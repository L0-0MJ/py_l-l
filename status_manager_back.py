import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from airflow.providers.amazon.aws.hooks.athena import AthenaHook

from dp.etl.document_processing.document_config import DocumentConfig, DocumentType
from dp.etl.document_processing.constants import ParameterKeys, ProcessingStatus
from dp.etl.utils.document_processing_utils import format_timestamp_for_athena, log_processing_step

logger = logging.getLogger(__name__)


class StatusManager:
    """문서 처리 상태 관리를 위한 중앙화된 클래스"""
    
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.athena_hook = AthenaHook(aws_conn_id="aws_default")
    
    def insert_init_status(self, inputfile_list_metadata: List[Dict[str, Any]], dag_info: Dict[str, str]) -> List[str]:
        """init_params 단계에서 INIT 상태로 레코드 삽입"""
        logger.info("=== Inserting INIT status records ===")
        
        inserted_document_ids = []
        current_timestamp = format_timestamp_for_athena(datetime.now())
        
        for file_metadata in inputfile_list_metadata:
            try:
                # 문서 타입 결정
                system = file_metadata.get(ParameterKeys.SYSTEM, "")
                doc_type = DocumentConfig.detect_document_type(system)
                param_keys = DocumentConfig.get_dynamic_param_keys(doc_type)
                
                # 문서 ID 생성
                document_id = self._generate_document_id(file_metadata)
                
                # 타겟 테이블 결정
                target_table = f"{self.params[param_keys['target_database_key']]}.{self.params[param_keys['target_table_key']]}"
                
                # INIT 상태 INSERT 쿼리 생성
                insert_query = self._build_init_status_query(
                    target_table=target_table,
                    document_id=document_id,
                    file_metadata=file_metadata,
                    dag_info=dag_info,
                    current_timestamp=current_timestamp
                )
                
                # 쿼리 실행
                query_execution_id = self.athena_hook.run_query(
                    query=insert_query,
                    query_context={"Database": self.params[param_keys["target_database_key"]]},
                    result_configuration={"OutputLocation": self.params["athena_output_location"]},
                )
                
                logger.info(f"INIT status inserted for document_id: {document_id}, execution_id: {query_execution_id}")
                inserted_document_ids.append(document_id)
                
                log_processing_step(
                    "INIT_STATUS_INSERTED",
                    document_id=document_id,
                    system=system,
                    document_type=doc_type.value,
                )
                
            except Exception as e:
                logger.error(f"Failed to insert INIT status for file {file_metadata.get(ParameterKeys.INPUT_FILE_PATH)}: {e}")
                continue
        
        logger.info(f"Successfully inserted INIT status for {len(inserted_document_ids)} documents")
        return inserted_document_ids
    
    def update_metadata_load_status(self, document_id: str, system: str) -> bool:
        """메타데이터 로드 완료 후 METADATA_LOAD 상태로 업데이트"""
        logger.info(f"=== Updating METADATA_LOAD status for document_id: {document_id} ===")
        
        try:
            # 문서 타입 결정
            doc_type = DocumentConfig.detect_document_type(system)
            param_keys = DocumentConfig.get_dynamic_param_keys(doc_type)
            
            # 타겟 테이블 결정
            target_database = self.params.get(param_keys['target_database_key'])
            target_table_name = self.params.get(param_keys['target_table_key'])
            target_table = f"{target_database}.{target_table_name}"
            
            # UPDATE 쿼리 생성
            current_timestamp = format_timestamp_for_athena(datetime.now())
            update_query = f"""
            UPDATE {target_table}
            SET 
                processing_status = '{ProcessingStatus.METADATA_LOAD}',
                last_processed_dtm = TIMESTAMP '{current_timestamp}',
                dp_mod_dtm = TIMESTAMP '{current_timestamp}'
            WHERE document_id = '{document_id}'
            """
            
            # 쿼리 실행
            query_execution_id = self.athena_hook.run_query(
                query=update_query,
                query_context={"Database": target_database},
                result_configuration={"OutputLocation": self.params["athena_output_location"]},
            )
            
            logger.info(f"METADATA_LOAD status updated for document_id: {document_id}, execution_id: {query_execution_id}")
            
            log_processing_step(
                "METADATA_LOAD_STATUS_UPDATED",
                document_id=document_id,
                system=system,
                document_type=doc_type.value,
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update METADATA_LOAD status for document_id {document_id}: {e}")
            return False

    def update_metadata_complete_status(self, document_id: str, system: str, metadata_result: Dict[str, Any]) -> bool:
        """정형메타 처리 완료 후 METADATA_COMPLETE 상태로 업데이트"""
        logger.info(f"=== Updating METADATA_COMPLETE status for document_id: {document_id} ===")
        
        try:
            # 문서 타입 결정
            doc_type = DocumentConfig.detect_document_type(system)
            param_keys = DocumentConfig.get_dynamic_param_keys(doc_type)
            
            # 타겟 테이블 결정
            target_table = f"{self.params[param_keys['target_database_key']]}.{self.params[param_keys['target_table_key']]}"
            
            # UPDATE 쿼리 생성
            current_timestamp = format_timestamp_for_athena(datetime.now())
            update_query = f"""
            UPDATE {target_table}
            SET 
                processing_status = '{ProcessingStatus.METADATA_COMPLETE}',
                last_processed_dtm = TIMESTAMP'{current_timestamp}',
                dp_mod_dtm = TIMESTAMP'{current_timestamp}'
            WHERE document_id = '{document_id}'
            """
            
            # 쿼리 실행
            query_execution_id = self.athena_hook.run_query(
                query=update_query,
                query_context={"Database": self.params[param_keys["target_database_key"]]},
                result_configuration={"OutputLocation": self.params["athena_output_location"]},
            )
            
            logger.info(f"METADATA_COMPLETE status updated for document_id: {document_id}, execution_id: {query_execution_id}")
            
            log_processing_step(
                "METADATA_COMPLETE_STATUS_UPDATED",
                document_id=document_id,
                system=system,
                document_type=doc_type.value,
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update METADATA_COMPLETE status for document_id {document_id}: {e}")
            return False
    
    def _generate_document_id(self, file_metadata: Dict[str, Any]) -> str:
        """파일 메타데이터를 기반으로 문서 ID 생성"""
        system = file_metadata.get(ParameterKeys.SYSTEM, "")
        file_id = file_metadata.get(ParameterKeys.FILE_ID, "")
        file_seq = file_metadata.get(ParameterKeys.FILE_SEQ, "")
        
        # 시스템별 고유 식별자 생성 방식
        if system == "MM":  # CONTRACT
            return f"CONTRACT_{file_id}_{file_seq}"
        elif system == "SM":  # LETTER
            return f"LETTER_{file_id}_{file_seq}"
        else:
            return f"DOC_{system}_{file_id}_{file_seq}"
    
    def _build_init_status_query(
        self, 
        target_table: str, 
        document_id: str, 
        file_metadata: Dict[str, Any], 
        dag_info: Dict[str, str],
        current_timestamp: str
    ) -> str:
        """INIT 상태 INSERT 쿼리 생성"""
        
        system = file_metadata.get(ParameterKeys.SYSTEM, "")
        file_id = file_metadata.get(ParameterKeys.FILE_ID, "")
        file_seq = file_metadata.get(ParameterKeys.FILE_SEQ, "")
        input_file_path = file_metadata.get(ParameterKeys.INPUT_FILE_PATH, "")
        parser_category = file_metadata.get(ParameterKeys.PARSER_CATEGORY, "")
        
        # 기본적인 컬럼들만 INSERT (최소 필수 정보)
        query = f"""
        INSERT INTO {target_table} (
            document_id,
            system,
            orig_file_id,
            orig_file_seq,
            file_path,
            parser_category,
            processing_status,
            last_processed_dtm,
            dag_id,
            dag_run_id,
            task_instance_id,
            dp_reg_dtm,
            dp_mod_dtm
        ) VALUES (
            '{document_id}',
            '{system}',
            '{file_id}',
            '{file_seq}',
            '{input_file_path}',
            '{parser_category}',
            '{ProcessingStatus.INIT}',
            TIMESTAMP'{current_timestamp}',
            '{dag_info["dag_id"]}',
            '{dag_info["dag_run_id"]}',
            '{dag_info.get("task_instance_id", "")}',
            TIMESTAMP'{current_timestamp}',
            TIMESTAMP'{current_timestamp}'
        )
        """
        
        return query


def insert_init_status_records(params: Dict[str, Any], inputfile_list_metadata: List[Dict[str, Any]], dag_info: Dict[str, str]) -> List[str]:
    """init_params에서 사용할 상태 삽입 함수"""
    status_manager = StatusManager(params)
    return status_manager.insert_init_status(inputfile_list_metadata, dag_info)


def update_metadata_complete_status(params: Dict[str, Any], document_id: str, system: str, metadata_result: Dict[str, Any]) -> bool:
    """메타데이터 처리 완료 후 사용할 상태 업데이트 함수"""
    status_manager = StatusManager(params)
    return status_manager.update_metadata_complete_status(document_id, system, metadata_result)