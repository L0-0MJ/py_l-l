# 1단계 문제 6: 로깅 데코레이터 - 해답

import logging
import functools
import time
import json
import inspect
import os
import uuid
import socket
import threading
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

# 전역 변수들
correlation_id_storage = threading.local()
sensitive_patterns = ["password", "token", "secret", "key", "credential", "auth"]

def get_correlation_id() -> str:
    """현재 스레드의 상관관계 ID를 반환합니다."""
    if not hasattr(correlation_id_storage, 'correlation_id'):
        correlation_id_storage.correlation_id = str(uuid.uuid4())
    return correlation_id_storage.correlation_id

def set_correlation_id(correlation_id: str) -> None:
    """현재 스레드의 상관관계 ID를 설정합니다."""
    correlation_id_storage.correlation_id = correlation_id

def mask_sensitive_data(data: Any, patterns: List[str] = None) -> Any:
    """
    민감한 데이터를 마스킹합니다.

    Args:
        data: 마스킹할 데이터
        patterns: 민감한 정보를 나타내는 패턴들

    Returns:
        마스킹된 데이터
    """
    if patterns is None:
        patterns = sensitive_patterns

    def _mask_value(value):
        """값 마스킹"""
        if isinstance(value, str) and len(value) > 0:
            if len(value) <= 4:
                return "*" * len(value)
            else:
                return value[:2] + "*" * (len(value) - 4) + value[-2:]
        return "***MASKED***"

    def _should_mask(key):
        """키가 마스킹 대상인지 확인"""
        if isinstance(key, str):
            return any(pattern.lower() in key.lower() for pattern in patterns)
        return False

    if isinstance(data, dict):
        return {
            key: _mask_value(value) if _should_mask(key) else mask_sensitive_data(value, patterns)
            for key, value in data.items()
        }
    elif isinstance(data, list):
        return [mask_sensitive_data(item, patterns) for item in data]
    elif isinstance(data, tuple):
        return tuple(mask_sensitive_data(item, patterns) for item in data)
    else:
        return data

def basic_logger(func: Callable) -> Callable:
    """
    기본 로깅 데코레이터

    함수 호출 전후에 기본적인 로그를 출력합니다.

    Args:
        func: 로깅할 함수

    Returns:
        래핑된 함수
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        start_time = time.time()
        start_datetime = datetime.now()

        # 함수 시그니처 정보 추출
        signature = inspect.signature(func)
        bound_args = signature.bind(*args, **kwargs)
        bound_args.apply_defaults()

        print(f"🔍 [{start_datetime.strftime('%Y-%m-%d %H:%M:%S')}] 함수 호출 시작: {func_name}")
        print(f"   📥 입력 파라미터: {dict(bound_args.arguments)}")

        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time

            print(f"   ✅ 함수 실행 완료: {func_name}")
            print(f"   📤 반환값: {result}")
            print(f"   ⏱️  실행 시간: {execution_time:.4f}초")

            return result

        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time

            print(f"   ❌ 함수 실행 실패: {func_name}")
            print(f"   🚫 에러: {type(e).__name__}: {str(e)}")
            print(f"   ⏱️  실행 시간: {execution_time:.4f}초")

            raise

    return wrapper

def advanced_logger(level: str = "INFO",
                   log_format: str = None,
                   log_to_file: bool = False,
                   filename: str = None,
                   mask_sensitive: bool = True):
    """
    고급 로깅 데코레이터

    로그 레벨, 포맷, 파일 출력 등을 설정할 수 있는 고급 로깅 기능을 제공합니다.

    Args:
        level: 로그 레벨
        log_format: 로그 포맷
        log_to_file: 파일 출력 여부
        filename: 로그 파일명
        mask_sensitive: 민감한 정보 마스킹 여부

    Returns:
        데코레이터 함수
    """
    def decorator(func: Callable) -> Callable:
        # 로거 설정
        logger = logging.getLogger(f"advanced_logger.{func.__name__}")
        logger.setLevel(getattr(logging, level.upper()))

        # 핸들러가 없으면 추가
        if not logger.handlers:
            # 콘솔 핸들러
            console_handler = create_console_handler(level)
            logger.addHandler(console_handler)

            # 파일 핸들러 (필요시)
            if log_to_file:
                file_handler = create_file_handler(filename or f"{func.__name__}.log", level)
                logger.addHandler(file_handler)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            start_time = time.time()

            # 파라미터 준비
            signature = inspect.signature(func)
            bound_args = signature.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # 민감한 정보 마스킹
            log_args = mask_sensitive_data(dict(bound_args.arguments)) if mask_sensitive else dict(bound_args.arguments)

            logger.info(f"🚀 Starting function: {func_name}")
            logger.debug(f"📥 Parameters: {log_args}")

            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                execution_time = end_time - start_time

                # 결과 마스킹
                log_result = mask_sensitive_data(result) if mask_sensitive else result

                logger.info(f"✅ Function completed: {func_name} ({execution_time:.4f}s)")
                logger.debug(f"📤 Result: {log_result}")

                return result

            except Exception as e:
                end_time = time.time()
                execution_time = end_time - start_time

                logger.error(f"❌ Function failed: {func_name} ({execution_time:.4f}s)")
                logger.error(f"🚫 Error: {type(e).__name__}: {str(e)}")

                # 중요한 에러는 WARNING 레벨로도 기록
                if isinstance(e, (PermissionError, SecurityException, ValueError)):
                    logger.warning(f"⚠️  Security/Validation error in {func_name}: {str(e)}")

                raise

        return wrapper
    return decorator

def structured_logger(include_metadata: bool = True,
                     include_performance: bool = True,
                     include_correlation_id: bool = True):
    """
    구조화된 로깅 데코레이터

    JSON 형태의 구조화된 로그를 출력합니다.

    Args:
        include_metadata: 메타데이터 포함 여부
        include_performance: 성능 정보 포함 여부
        include_correlation_id: 상관관계 ID 포함 여부

    Returns:
        데코레이터 함수
    """
    def decorator(func: Callable) -> Callable:
        logger = logging.getLogger(f"structured_logger.{func.__name__}")
        logger.setLevel(logging.INFO)

        # JSON 포맷터 설정
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(handler)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            start_time = time.time()
            start_datetime = datetime.now()

            # 기본 로그 구조
            log_entry = {
                "timestamp": start_datetime.isoformat(),
                "event": "function_call",
                "function": func_name,
                "status": "started"
            }

            # 상관관계 ID
            if include_correlation_id:
                log_entry["correlation_id"] = get_correlation_id()

            # 메타데이터
            if include_metadata:
                log_entry["metadata"] = {
                    "hostname": socket.gethostname(),
                    "process_id": os.getpid(),
                    "thread_id": threading.get_ident(),
                    "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}"
                }

            # 파라미터 정보
            signature = inspect.signature(func)
            bound_args = signature.bind(*args, **kwargs)
            bound_args.apply_defaults()
            log_entry["parameters"] = mask_sensitive_data(dict(bound_args.arguments))

            # 시작 로그
            logger.info(f"📊 STRUCTURED_LOG: {json.dumps(log_entry, ensure_ascii=False)}")

            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                execution_time = end_time - start_time

                # 완료 로그
                completion_entry = log_entry.copy()
                completion_entry.update({
                    "status": "completed",
                    "timestamp": datetime.now().isoformat(),
                    "result": mask_sensitive_data(result)
                })

                # 성능 정보
                if include_performance:
                    completion_entry["performance"] = {
                        "execution_time_seconds": execution_time,
                        "start_time": start_datetime.isoformat(),
                        "end_time": datetime.now().isoformat()
                    }

                logger.info(f"📊 STRUCTURED_LOG: {json.dumps(completion_entry, ensure_ascii=False)}")

                return result

            except Exception as e:
                end_time = time.time()
                execution_time = end_time - start_time

                # 에러 로그
                error_entry = log_entry.copy()
                error_entry.update({
                    "status": "error",
                    "timestamp": datetime.now().isoformat(),
                    "error": {
                        "type": type(e).__name__,
                        "message": str(e),
                        "execution_time_seconds": execution_time
                    }
                })

                logger.error(f"📊 STRUCTURED_LOG: {json.dumps(error_entry, ensure_ascii=False)}")

                raise

        return wrapper
    return decorator

class SecurityException(Exception):
    """보안 관련 예외"""
    pass

def audit_logger(user_context: Dict[str, Any] = None,
                session_id: str = None,
                track_changes: bool = True,
                security_event: bool = False):
    """
    감사 로깅 데코레이터

    보안 및 감사를 위한 상세한 로그를 기록합니다.

    Args:
        user_context: 사용자 컨텍스트 정보
        session_id: 세션 ID
        track_changes: 변경사항 추적 여부
        security_event: 보안 이벤트 여부

    Returns:
        데코레이터 함수
    """
    def decorator(func: Callable) -> Callable:
        logger = logging.getLogger(f"audit_logger.{func.__name__}")
        logger.setLevel(logging.INFO)

        # 감사 로그는 항상 파일에 저장
        if not logger.handlers:
            # 감사 로그 파일 핸들러
            audit_handler = logging.FileHandler("audit.log", encoding='utf-8')
            audit_formatter = logging.Formatter(
                '[%(asctime)s] AUDIT - %(name)s - %(levelname)s - %(message)s'
            )
            audit_handler.setFormatter(audit_formatter)
            logger.addHandler(audit_handler)

            # 보안 이벤트는 별도 파일에도 저장
            if security_event:
                security_handler = logging.FileHandler("security.log", encoding='utf-8')
                security_formatter = logging.Formatter(
                    '[%(asctime)s] SECURITY - %(name)s - %(levelname)s - %(message)s'
                )
                security_handler.setFormatter(security_formatter)
                logger.addHandler(security_handler)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            start_time = time.time()
            audit_id = str(uuid.uuid4())

            # 감사 로그 엔트리 생성
            audit_entry = {
                "audit_id": audit_id,
                "function": func_name,
                "timestamp": datetime.now().isoformat(),
                "correlation_id": get_correlation_id(),
                "user_context": user_context or {"user": "system"},
                "session_id": session_id or "no_session",
                "security_event": security_event
            }

            # 파라미터 정보 (민감한 정보는 마스킹하되, 감사를 위해 더 보수적으로)
            signature = inspect.signature(func)
            bound_args = signature.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # 감사용으로 파라미터 정보를 더 상세히 기록
            audit_entry["parameters"] = {}
            for key, value in bound_args.arguments.items():
                if any(pattern.lower() in key.lower() for pattern in sensitive_patterns):
                    audit_entry["parameters"][key] = "***REDACTED***"
                else:
                    audit_entry["parameters"][key] = str(value)[:100]  # 길이 제한

            # 시작 감사 로그
            logger.info(f"🔐 AUDIT START: {json.dumps(audit_entry, ensure_ascii=False)}")

            # 보안 이벤트라면 추가 로그
            if security_event:
                logger.warning(f"🚨 SECURITY EVENT: {func_name} called with audit_id: {audit_id}")

            try:
                # 변경사항 추적을 위한 이전 상태 캡처 (예시)
                before_state = None
                if track_changes and hasattr(args[0] if args else None, '__dict__'):
                    before_state = getattr(args[0], '__dict__', {}).copy()

                result = func(*args, **kwargs)

                end_time = time.time()
                execution_time = end_time - start_time

                # 완료 감사 로그
                completion_entry = audit_entry.copy()
                completion_entry.update({
                    "status": "success",
                    "execution_time_seconds": execution_time,
                    "end_timestamp": datetime.now().isoformat()
                })

                # 결과 정보 (민감하지 않은 경우만)
                if result is not None:
                    result_str = str(result)[:200]  # 결과 길이 제한
                    if not any(pattern.lower() in result_str.lower() for pattern in sensitive_patterns):
                        completion_entry["result_preview"] = result_str
                    else:
                        completion_entry["result_preview"] = "***SENSITIVE_DATA***"

                # 변경사항 추적
                if track_changes and before_state is not None:
                    if hasattr(args[0] if args else None, '__dict__'):
                        after_state = getattr(args[0], '__dict__', {})
                        changes = {}
                        for key in set(before_state.keys()) | set(after_state.keys()):
                            before_val = before_state.get(key, "<MISSING>")
                            after_val = after_state.get(key, "<MISSING>")
                            if before_val != after_val:
                                changes[key] = {
                                    "before": str(before_val)[:50],
                                    "after": str(after_val)[:50]
                                }
                        if changes:
                            completion_entry["data_changes"] = changes

                logger.info(f"🔐 AUDIT COMPLETE: {json.dumps(completion_entry, ensure_ascii=False)}")

                return result

            except Exception as e:
                end_time = time.time()
                execution_time = end_time - start_time

                # 에러 감사 로그
                error_entry = audit_entry.copy()
                error_entry.update({
                    "status": "error",
                    "execution_time_seconds": execution_time,
                    "end_timestamp": datetime.now().isoformat(),
                    "error": {
                        "type": type(e).__name__,
                        "message": str(e)[:500]  # 에러 메시지 길이 제한
                    }
                })

                logger.error(f"🔐 AUDIT ERROR: {json.dumps(error_entry, ensure_ascii=False)}")

                # 보안 관련 에러라면 특별히 기록
                if isinstance(e, (PermissionError, SecurityException)) or security_event:
                    logger.critical(f"🚨 SECURITY FAILURE: {func_name} - {str(e)} (audit_id: {audit_id})")

                raise

        return wrapper
    return decorator

# 로그 설정 함수들
def setup_logging():
    """로깅 환경을 설정합니다."""
    # 루트 로거 설정
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

    # 디렉토리 생성
    os.makedirs('logs', exist_ok=True)

def create_file_handler(filename: str, level: str) -> logging.Handler:
    """파일 핸들러를 생성합니다."""
    os.makedirs('logs', exist_ok=True)
    filepath = os.path.join('logs', filename)

    handler = logging.FileHandler(filepath, encoding='utf-8')
    handler.setLevel(getattr(logging, level.upper()))

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)

    return handler

def create_console_handler(level: str) -> logging.Handler:
    """콘솔 핸들러를 생성합니다."""
    handler = logging.StreamHandler()
    handler.setLevel(getattr(logging, level.upper()))

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)

    return handler

# 테스트용 클래스와 함수들
class UserService:
    """사용자 서비스 테스트 클래스"""

    def __init__(self):
        self.users_db = {
            "user1": {"name": "Alice", "email": "alice@example.com", "password": "secret123"},
            "user2": {"name": "Bob", "email": "bob@example.com", "password": "password456"}
        }

    @basic_logger
    def get_user(self, user_id: str):
        """사용자 정보 조회"""
        time.sleep(0.1)  # DB 조회 시뮬레이션
        return self.users_db.get(user_id, None)

    @advanced_logger(level="INFO", mask_sensitive=True)
    def create_user(self, name: str, email: str, password: str):
        """새 사용자 생성"""
        time.sleep(0.2)
        user_id = f"user_{len(self.users_db) + 1}"
        self.users_db[user_id] = {"name": name, "email": email, "password": password}
        return user_id

    @structured_logger(include_metadata=True, include_performance=True)
    def update_user(self, user_id: str, **updates):
        """사용자 정보 업데이트"""
        time.sleep(0.15)
        if user_id in self.users_db:
            self.users_db[user_id].update(updates)
            return True
        return False

    @audit_logger(security_event=True, track_changes=True)
    def delete_user(self, user_id: str, admin_user: str):
        """사용자 삭제 (관리자 권한 필요)"""
        time.sleep(0.1)
        if user_id in self.users_db:
            deleted_user = self.users_db.pop(user_id)
            return {"deleted": True, "user": deleted_user}
        return {"deleted": False, "reason": "User not found"}

@basic_logger
def simple_calculation(x: int, y: int, operation: str = "add"):
    """간단한 계산 함수"""
    time.sleep(0.05)
    if operation == "add":
        return x + y
    elif operation == "multiply":
        return x * y
    elif operation == "divide":
        return x / y if y != 0 else None
    else:
        raise ValueError(f"Unknown operation: {operation}")

@advanced_logger(level="DEBUG", log_to_file=True, filename="api_calls.log")
def api_call_simulation(endpoint: str, data: Dict[str, Any], api_key: str):
    """API 호출 시뮬레이션"""
    time.sleep(0.3)
    if not api_key:
        raise ValueError("API key is required")
    return {"status": "success", "endpoint": endpoint, "response": "Mock API response"}

@structured_logger(include_metadata=True, include_correlation_id=True)
def data_processing_pipeline(data: List[Dict[str, Any]], config: Dict[str, Any]):
    """데이터 처리 파이프라인"""
    time.sleep(0.4)
    processed_count = 0
    for item in data:
        # 데이터 처리 로직 시뮬레이션
        if item.get("valid", True):
            processed_count += 1

    return {
        "total_items": len(data),
        "processed_items": processed_count,
        "config_used": config
    }

@audit_logger(security_event=True)
def admin_operation(operation: str, target: str, admin_user: str, admin_token: str):
    """관리자 작업"""
    time.sleep(0.2)
    if admin_token != "admin_secret_token":
        raise PermissionError("Invalid admin token")

    return f"Admin operation '{operation}' performed on '{target}' by {admin_user}"

# 복합 데코레이터 테스트
@audit_logger(user_context={"user": "system", "role": "admin"}, security_event=True)
@structured_logger(include_metadata=True, include_performance=True)
def critical_system_operation(action: str, parameters: Dict[str, Any]):
    """중요한 시스템 작업 (여러 데코레이터 조합)"""
    time.sleep(0.5)
    if action == "shutdown":
        return {"status": "system_shutdown_initiated", "timestamp": datetime.now().isoformat()}
    return {"status": f"action_{action}_completed", "parameters": parameters}

# 테스트 코드
if __name__ == "__main__":
    print("=== 1단계 문제 6: 로깅 데코레이터 테스트 ===\n")

    # 로깅 환경 설정
    setup_logging()

    print("--- 기본 로거 테스트 ---")
    result = simple_calculation(10, 5, "multiply")
    print(f"계산 결과: {result}")
    print()

    print("--- 고급 로거 테스트 ---")
    try:
        result = api_call_simulation(
            endpoint="/users",
            data={"name": "Test User"},
            api_key="secret_api_key_12345"
        )
        print(f"API 결과: {result}")
    except Exception as e:
        print(f"API 에러: {e}")
    print()

    print("--- 구조화된 로거 테스트 ---")
    set_correlation_id(str(uuid.uuid4()))

    sample_data = [
        {"id": 1, "value": "data1", "valid": True},
        {"id": 2, "value": "data2", "valid": False},
        {"id": 3, "value": "data3", "valid": True}
    ]

    config = {"batch_size": 100, "timeout": 30, "api_key": "secret_config_key"}
    result = data_processing_pipeline(sample_data, config)
    print(f"파이프라인 결과: {result}")
    print()

    print("--- UserService 테스트 ---")
    user_service = UserService()

    # 사용자 조회
    user = user_service.get_user("user1")
    print(f"조회된 사용자: {user['name'] if user else 'Not found'}")

    # 사용자 생성
    new_user_id = user_service.create_user("Charlie", "charlie@example.com", "mypassword789")
    print(f"새 사용자 ID: {new_user_id}")

    # 사용자 정보 업데이트
    update_success = user_service.update_user(new_user_id, name="Charles", age=30)
    print(f"업데이트 성공: {update_success}")

    # 감사 로그가 필요한 작업
    print("\n--- 감사 로거 테스트 ---")
    try:
        delete_result = user_service.delete_user(new_user_id, "admin_user")
        print(f"삭제 결과: {delete_result}")
    except Exception as e:
        print(f"삭제 에러: {e}")

    # 관리자 작업
    try:
        admin_result = admin_operation("USER_DELETE", "user123", "admin", "admin_secret_token")
        print(f"관리자 작업 결과: {admin_result}")
    except Exception as e:
        print(f"관리자 작업 에러: {e}")

    print("\n--- 복합 데코레이터 테스트 ---")
    try:
        critical_result = critical_system_operation(
            "maintenance",
            {"target": "database", "duration": "30min", "admin_password": "secret123"}
        )
        print(f"중요 작업 결과: {critical_result}")
    except Exception as e:
        print(f"중요 작업 에러: {e}")

    print("\n--- 에러 상황 테스트 ---")
    try:
        result = simple_calculation(10, 0, "divide")
        print(f"나눗셈 결과: {result}")
    except Exception as e:
        print(f"나눗셈 에러: {e}")

    try:
        admin_result = admin_operation("SYSTEM_RESET", "production", "hacker", "wrong_token")
    except Exception as e:
        print(f"보안 에러: {e}")

    print("\n--- 민감한 정보 마스킹 테스트 ---")
    sensitive_data = {
        "username": "alice",
        "password": "secret123",
        "api_key": "sk-abcd1234567890",
        "normal_data": "public_info",
        "token": "bearer_xyz789",
        "email": "alice@example.com"
    }

    masked = mask_sensitive_data(sensitive_data)
    print(f"원본 데이터: {sensitive_data}")
    print(f"마스킹 데이터: {masked}")

    print("\n=== 로깅 데코레이터 테스트 완료 ===")
    print("🎯 학습한 로깅 패턴:")
    print("  - 기본 로깅 (함수 호출 추적)")
    print("  - 고급 로깅 (레벨, 파일 출력, 민감정보 마스킹)")
    print("  - 구조화된 로깅 (JSON, 메타데이터, 상관관계 ID)")
    print("  - 감사 로깅 (보안, 변경사항 추적)")
    print("  - 복합 데코레이터 (여러 로깅 전략 조합)")
    print("  - 민감한 정보 마스킹")
    print("  - 에러 상황 로깅")

    print(f"\n📁 로그 파일 생성 위치: ./logs/ 디렉토리")
    print(f"📋 감사 로그: audit.log")
    print(f"🔐 보안 로그: security.log")
    print(f"🔧 API 로그: logs/api_calls.log")

"""
학습 포인트:
1. Python logging 모듈의 전문적 사용법
2. 다양한 로그 레벨과 핸들러 활용
3. 구조화된 로깅 (JSON 형태)
4. 민감한 정보 마스킹 기법
5. 상관관계 ID를 통한 분산 추적
6. 감사 로그와 보안 로그의 중요성
7. 메타데이터와 성능 지표 수집
8. 여러 데코레이터의 조합 사용법
9. 스레드 안전한 로깅 구현
10. 파일과 콘솔 동시 출력

실제 AI Agent 시스템에서는 이러한 로깅이 필수적입니다:
- API 호출 추적
- 사용자 행동 분석
- 보안 이벤트 모니터링
- 성능 병목 지점 파악
- 디버깅 및 트러블슈팅
- 컴플라이언스 요구사항 충족
"""