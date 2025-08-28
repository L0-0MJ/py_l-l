"""
데이터 검증 시스템 - 1단계 프로젝트
데이터 품질 검증 라이브러리 구현

주요 기능:
- CSV 데이터 검증
- 성능 모니터링
- 커스텀 예외 처리
- 안전한 파일 처리
"""

import pandas as pd
import json
import time
import psutil
import os
from typing import Dict, Any, List, Optional
from functools import wraps
from contextlib import contextmanager


# 1. 커스텀 예외 클래스들
class DataValidationError(Exception):
    """데이터 검증 실패 시 발생하는 예외"""
    pass


class ConfigurationError(Exception):
    """설정 파일 관련 예외"""
    pass


class FileProcessingError(Exception):
    """파일 처리 관련 예외"""
    pass


# 2. 성능 측정 데코레이터
def performance_monitor(func):
    """함수 실행 시간 및 메모리 사용량 측정 데코레이터"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 시작 시간 및 메모리
        start_time = time.time()
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            result = func(*args, **kwargs)
            
            # 종료 시간 및 메모리
            end_time = time.time()
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # 성능 정보 출력
            execution_time = end_time - start_time
            memory_used = end_memory - start_memory
            
            print(f"\n📊 [{func.__name__}] 성능 리포트:")
            print(f"   ⏱️  실행시간: {execution_time:.2f}초")
            print(f"   💾 메모리 사용: {memory_used:.2f}MB")
            print(f"   📈 최대 메모리: {end_memory:.2f}MB")
            
            return result
            
        except Exception as e:
            end_time = time.time()
            print(f"\n❌ [{func.__name__}] 실행 실패:")
            print(f"   ⏱️  실행시간: {end_time - start_time:.2f}초")
            print(f"   🚫 에러: {str(e)}")
            raise
    
    return wrapper


# 3. 파일 안전 처리 컨텍스트 매니저
@contextmanager
def safe_file_handler(file_path: str, mode: str = 'r', encoding: str = 'utf-8'):
    """안전한 파일 처리를 위한 컨텍스트 매니저"""
    file_obj = None
    try:
        if not os.path.exists(file_path) and 'r' in mode:
            raise FileProcessingError(f"파일을 찾을 수 없습니다: {file_path}")
        
        file_obj = open(file_path, mode, encoding=encoding)
        print(f"✅ 파일 열기 성공: {file_path}")
        yield file_obj
        
    except IOError as e:
        raise FileProcessingError(f"파일 처리 중 오류 발생: {str(e)}")
    finally:
        if file_obj:
            file_obj.close()
            print(f"🔒 파일 안전하게 닫음: {file_path}")


# 4. 메인 데이터 검증 클래스
class DataValidator:
    """데이터 품질 검증을 위한 메인 클래스"""
    
    def __init__(self, config_path: str):
        """
        DataValidator 초기화
        
        Args:
            config_path: 검증 규칙이 정의된 JSON 설정 파일 경로
        """
        self.config_path = config_path
        self._validation_rules = None
        self._load_config()
    
    def _load_config(self):
        """설정 파일 로드"""
        try:
            with safe_file_handler(self.config_path, 'r') as f:
                self._validation_rules = json.load(f)
            print(f"📋 설정 파일 로드 완료: {len(self._validation_rules)} 개 규칙")
        except (FileProcessingError, json.JSONDecodeError) as e:
            raise ConfigurationError(f"설정 파일 로드 실패: {str(e)}")
    
    @property
    def validation_rules(self) -> Dict[str, Any]:
        """검증 규칙 반환 (읽기 전용)"""
        return self._validation_rules.copy() if self._validation_rules else {}
    
    @performance_monitor
    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        데이터셋 전체 검증 수행
        
        Args:
            df: 검증할 pandas DataFrame
            
        Returns:
            검증 결과 리포트 딕셔너리
        """
        if df.empty:
            raise DataValidationError("빈 데이터셋은 검증할 수 없습니다")
        
        report = {
            "총_행수": len(df),
            "총_열수": len(df.columns),
            "검증_통과": True,
            "오류_목록": [],
            "경고_목록": [],
            "컬럼별_리포트": {}
        }
        
        # 각 컬럼별 검증 수행
        for column in df.columns:
            column_report = self._validate_column(df, column)
            report["컬럼별_리포트"][column] = column_report
            
            if column_report["오류_개수"] > 0:
                report["검증_통과"] = False
                report["오류_목록"].extend(column_report["오류_목록"])
            
            if column_report["경고_개수"] > 0:
                report["경고_목록"].extend(column_report["경고_목록"])
        
        return report
    
    def _validate_column(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """개별 컬럼 검증"""
        series = df[column]
        rules = self._validation_rules.get(column, {})
        
        column_report = {
            "데이터_타입": str(series.dtype),
            "null_개수": series.isnull().sum(),
            "유니크_값_개수": series.nunique(),
            "오류_개수": 0,
            "경고_개수": 0,
            "오류_목록": [],
            "경고_목록": []
        }
        
        # Null 값 검증
        if rules.get("required", False) and column_report["null_개수"] > 0:
            error_msg = f"{column}: 필수 컬럼에 {column_report['null_개수']}개의 null 값 존재"
            column_report["오류_목록"].append(error_msg)
            column_report["오류_개수"] += 1
        
        # 데이터 타입 검증
        expected_type = rules.get("type")
        if expected_type and not self._check_data_type(series, expected_type):
            error_msg = f"{column}: 예상 타입 {expected_type}, 실제 타입 {series.dtype}"
            column_report["오류_목록"].append(error_msg)
            column_report["오류_개수"] += 1
        
        # 범위 검증 (숫자형 데이터)
        if series.dtype in ['int64', 'float64'] and 'range' in rules:
            min_val, max_val = rules['range']
            out_of_range = series[(series < min_val) | (series > max_val)].count()
            if out_of_range > 0:
                warning_msg = f"{column}: {out_of_range}개 값이 범위 [{min_val}, {max_val}]를 벗어남"
                column_report["경고_목록"].append(warning_msg)
                column_report["경고_개수"] += 1
        
        return column_report
    
    def _check_data_type(self, series: pd.Series, expected_type: str) -> bool:
        """데이터 타입 검증 헬퍼"""
        type_mapping = {
            'integer': series.dtype in ['int64', 'int32'],
            'float': series.dtype in ['float64', 'float32'],
            'string': series.dtype == 'object',
            'boolean': series.dtype == 'bool'
        }
        return type_mapping.get(expected_type, True)
    
    @performance_monitor
    def save_report(self, report: Dict[str, Any], output_path: str):
        """검증 리포트를 JSON 파일로 저장"""
        try:
            with safe_file_handler(output_path, 'w') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"💾 리포트 저장 완료: {output_path}")
        except FileProcessingError as e:
            raise DataValidationError(f"리포트 저장 실패: {str(e)}")


# 5. 대용량 데이터 처리 함수
@performance_monitor
def process_large_dataset(file_path: str, chunk_size: int = 10000) -> Dict[str, Any]:
    """
    대용량 데이터셋을 청크 단위로 처리
    
    Args:
        file_path: 처리할 CSV 파일 경로
        chunk_size: 청크 크기
        
    Returns:
        처리 결과 요약
    """
    summary = {
        "총_처리_행수": 0,
        "청크_개수": 0,
        "처리_시간": 0,
        "에러_개수": 0
    }
    
    try:
        # CSV 파일을 청크 단위로 읽기
        chunk_iter = pd.read_csv(file_path, chunksize=chunk_size)
        
        for chunk_idx, chunk in enumerate(chunk_iter):
            try:
                # 각 청크 처리 (예: 기본적인 정제 작업)
                processed_chunk = chunk.dropna().drop_duplicates()
                
                summary["총_처리_행수"] += len(processed_chunk)
                summary["청크_개수"] += 1
                
                print(f"✅ 청크 {chunk_idx + 1} 처리 완료: {len(processed_chunk)} 행")
                
            except Exception as e:
                summary["에러_개수"] += 1
                print(f"❌ 청크 {chunk_idx + 1} 처리 실패: {str(e)}")
        
        return summary
        
    except FileNotFoundError:
        raise FileProcessingError(f"파일을 찾을 수 없습니다: {file_path}")
    except Exception as e:
        raise DataValidationError(f"데이터 처리 중 오류: {str(e)}")


if __name__ == "__main__":
    # 테스트 코드
    print("🚀 데이터 검증 시스템 테스트 시작")
    
    # 테스트용 설정 파일 생성
    config = {
        "name": {"required": True, "type": "string"},
        "age": {"required": True, "type": "integer", "range": [0, 120]},
        "salary": {"required": False, "type": "float", "range": [0, 1000000]}
    }
    
    config_path = "/Users/mzc03-minjeong/Documents/l&l/py/validation_config.json"
    with safe_file_handler(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # 테스트용 데이터 생성
    test_data = pd.DataFrame({
        'name': ['Alice', 'Bob', None, 'Diana'],
        'age': [25, 30, 35, 150],  # 150은 범위 초과
        'salary': [50000, 60000, 70000, 80000]
    })
    
    # DataValidator 테스트
    try:
        validator = DataValidator(config_path)
        report = validator.validate_dataset(test_data)
        
        print("\n📋 검증 리포트:")
        print(f"검증 통과: {'✅' if report['검증_통과'] else '❌'}")
        print(f"총 오류: {len(report['오류_목록'])}개")
        print(f"총 경고: {len(report['경고_목록'])}개")
        
        # 리포트 저장
        report_path = "/Users/mzc03-minjeong/Documents/l&l/py/validation_report.json"
        validator.save_report(report, report_path)
        
    except Exception as e:
        print(f"❌ 테스트 실패: {str(e)}")