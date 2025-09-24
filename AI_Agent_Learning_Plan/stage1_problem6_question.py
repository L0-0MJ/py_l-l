# 1단계 문제 6: 로깅 데코레이터
# 함수 호출 전후에 로그를 남기는 데코레이터를 만들어보세요.

"""
문제 요구사항:
1. basic_logger 데코레이터를 만드세요:
   - 함수 호출 전후에 기본 로그 출력
   - 함수 이름, 호출 시간, 실행 시간 기록
   - 입력 파라미터와 반환값 로깅

2. advanced_logger 데코레이터를 만드세요:
   - 로그 레벨 설정 가능 (DEBUG, INFO, WARNING, ERROR)
   - 로그 포맷 커스터마이징 가능
   - 파일 출력 옵션
   - 민감한 정보 마스킹 기능

3. structured_logger 데코레이터를 만드세요:
   - JSON 형태의 구조화된 로그
   - 메타데이터 추가 (호스트명, 프로세스 ID 등)
   - 상관관계 ID (correlation ID) 지원
   - 성능 메트릭 포함

4. audit_logger 데코레이터를 만드세요:
   - 감사용 상세 로그
   - 사용자 정보, 세션 정보 기록
   - 데이터 변경사항 추적
   - 보안 이벤트 로깅

힌트:
- logging 모듈을 사용하여 전문적인 로깅 구현
- datetime을 사용하여 정확한 타임스탬프 생성
- json 모듈을 사용하여 구조화된 로그 생성
- inspect 모듈을 사용하여 함수 시그니처 정보 추출
- 환경변수나 설정을 통한 로그 레벨 제어
- 민감한 정보 필터링을 위한 정규표현식 사용
"""

import logging
import functools
import time
import json
import inspect
import os
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union
import threading

# 전역 변수들
correlation_id_storage = threading.local()
sensitive_patterns = ["password", "token", "secret", "key", "credential"]

def get_correlation_id() -> str:
    """현재 스레드의 상관관계 ID를 반환합니다."""
    pass

def set_correlation_id(correlation_id: str) -> None:
    """현재 스레드의 상관관계 ID를 설정합니다."""
    pass

def mask_sensitive_data(data: Any, patterns: List[str] = None) -> Any:
    """민감한 데이터를 마스킹합니다."""
    pass

def basic_logger(func: Callable) -> Callable:
    """
    기본 로깅 데코레이터
    여기에 코드를 작성하세요.
    """
    pass

def advanced_logger(level: str = "INFO",
                   log_format: str = None,
                   log_to_file: bool = False,
                   filename: str = None,
                   mask_sensitive: bool = True):
    """
    고급 로깅 데코레이터
    여기에 코드를 작성하세요.
    """
    pass

def structured_logger(include_metadata: bool = True,
                     include_performance: bool = True,
                     include_correlation_id: bool = True):
    """
    구조화된 로깅 데코레이터
    여기에 코드를 작성하세요.
    """
    pass

def audit_logger(user_context: Dict[str, Any] = None,
                session_id: str = None,
                track_changes: bool = True,
                security_event: bool = False):
    """
    감사 로깅 데코레이터
    여기에 코드를 작성하세요.
    """
    pass

# 로그 설정 함수들
def setup_logging():
    """로깅 환경을 설정합니다."""
    pass

def create_file_handler(filename: str, level: str) -> logging.Handler:
    """파일 핸들러를 생성합니다."""
    pass

def create_console_handler(level: str) -> logging.Handler:
    """콘솔 핸들러를 생성합니다."""
    pass

# 테스트용 클래스와 함수들
class UserService:
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

    config = {"batch_size": 100, "timeout": 30}
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

    print("\n=== 로깅 데코레이터 테스트 완료 ===")
    print("다양한 로깅 패턴을 학습했습니다:")
    print("- 기본 로깅 (함수 호출 추적)")
    print("- 고급 로깅 (레벨, 파일 출력, 민감정보 마스킹)")
    print("- 구조화된 로깅 (JSON, 메타데이터, 상관관계 ID)")
    print("- 감사 로깅 (보안, 변경사항 추적)")
    print("- 에러 상황 로깅")