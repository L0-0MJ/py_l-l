# 1단계 문제 5: 에러 처리 데코레이터
# API 호출 실패 시 재시도하는 데코레이터를 만들어보세요.

"""
문제 요구사항:
1. retry_decorator 함수를 만드세요:
   - 최대 재시도 횟수 설정 가능 (기본값: 3)
   - 재시도 간격 설정 가능 (기본값: 1초)
   - 특정 예외 타입만 재시도 (기본값: 모든 예외)
   - 재시도할 때마다 로그 출력

2. exponential_backoff_retry 데코레이터를 만드세요:
   - 지수 백오프 구현 (재시도마다 대기 시간이 2배씩 증가)
   - 최대 대기 시간 제한 (기본값: 60초)
   - 백오프 계수 설정 가능 (기본값: 2)
   - 재시도 전 대기 시간 로그 출력

3. circuit_breaker 데코레이터를 만드세요:
   - 연속 실패 임계값 설정 (기본값: 5)
   - 임계값 도달 시 일정 시간 동안 함수 호출 차단
   - 차단 시간 경과 후 반열림 상태로 전환
   - 상태 변화 시 로그 출력

4. fallback_decorator를 만드세요:
   - 함수 실행 실패 시 대체 함수 호출
   - 대체 함수도 실패하면 기본값 반환
   - 어떤 방법으로 결과를 얻었는지 로그 출력

힌트:
- time.sleep()을 사용하여 재시도 간격 구현
- 예외 타입 검사: isinstance(exception, exception_types)
- 클래스나 딕셔너리를 사용하여 상태 관리
- datetime을 사용하여 시간 기반 로직 구현
- functools.wraps로 메타데이터 보존
"""

import time
import functools
import random
from datetime import datetime, timedelta
from typing import Any, Callable, Tuple, Type, Union, Optional, Dict

class CircuitBreakerState:
    """서킷 브레이커 상태 관리 클래스"""
    pass

# 전역 서킷 브레이커 상태 저장
circuit_breaker_states: Dict[str, CircuitBreakerState] = {}

def retry_decorator(max_retries: int = 3, delay: float = 1.0,
                   exceptions: Tuple[Type[Exception], ...] = (Exception,)):
    """
    기본 재시도 데코레이터
    여기에 코드를 작성하세요.
    """
    pass

def exponential_backoff_retry(max_retries: int = 3, initial_delay: float = 1.0,
                            backoff_factor: float = 2.0, max_delay: float = 60.0,
                            exceptions: Tuple[Type[Exception], ...] = (Exception,)):
    """
    지수 백오프 재시도 데코레이터
    여기에 코드를 작성하세요.
    """
    pass

def circuit_breaker(failure_threshold: int = 5, recovery_timeout: float = 60.0,
                   expected_exception: Type[Exception] = Exception):
    """
    서킷 브레이커 데코레이터
    여기에 코드를 작성하세요.
    """
    pass

def fallback_decorator(fallback_func: Callable, default_value: Any = None):
    """
    폴백 데코레이터
    여기에 코드를 작성하세요.
    """
    pass

# 테스트용 함수들
def unreliable_api_call(success_rate: float = 0.3, response_time: float = 0.1):
    """
    불안정한 API 호출을 시뮬레이션하는 함수
    success_rate: 성공 확률 (0.0 ~ 1.0)
    response_time: 응답 시간
    """
    time.sleep(response_time)
    if random.random() < success_rate:
        return f"API 호출 성공! 시간: {datetime.now().strftime('%H:%M:%S')}"
    else:
        raise ConnectionError("API 서버 연결 실패")

def database_query(success_rate: float = 0.4):
    """데이터베이스 쿼리 시뮬레이션"""
    time.sleep(0.2)
    if random.random() < success_rate:
        return "데이터베이스 쿼리 결과: [1, 2, 3, 4, 5]"
    else:
        raise TimeoutError("데이터베이스 연결 타임아웃")

def external_service_call():
    """외부 서비스 호출 시뮬레이션"""
    time.sleep(0.1)
    if random.random() < 0.2:  # 20% 성공률
        return "외부 서비스 응답 데이터"
    else:
        raise RuntimeError("외부 서비스 오류")

# 폴백 함수들
def api_fallback():
    """API 호출 실패 시 대체 함수"""
    time.sleep(0.1)
    if random.random() < 0.8:  # 80% 성공률
        return "캐시된 API 응답"
    else:
        raise Exception("폴백도 실패")

def db_fallback():
    """데이터베이스 쿼리 실패 시 대체 함수"""
    return "기본 데이터: [0, 0, 0]"

# 데코레이터가 적용된 테스트 함수들
@retry_decorator(max_retries=3, delay=0.5, exceptions=(ConnectionError,))
def test_retry_api():
    return unreliable_api_call(success_rate=0.4)

@exponential_backoff_retry(max_retries=4, initial_delay=0.2, backoff_factor=2.0, max_delay=5.0)
def test_backoff_db():
    return database_query(success_rate=0.3)

@circuit_breaker(failure_threshold=3, recovery_timeout=10.0, expected_exception=RuntimeError)
def test_circuit_breaker():
    return external_service_call()

@fallback_decorator(fallback_func=api_fallback, default_value="기본 응답")
def test_fallback():
    return unreliable_api_call(success_rate=0.1)

# 여러 데코레이터 조합 테스트
@fallback_decorator(fallback_func=db_fallback, default_value="최종 기본값")
@retry_decorator(max_retries=2, delay=0.3, exceptions=(TimeoutError,))
def test_combined_decorators():
    return database_query(success_rate=0.2)

# 테스트 코드
if __name__ == "__main__":
    print("=== 1단계 문제 5: 에러 처리 데코레이터 테스트 ===\n")

    # random 시드 설정 (재현 가능한 테스트를 위해)
    random.seed(42)

    print("--- 기본 재시도 데코레이터 테스트 ---")
    try:
        result = test_retry_api()
        print(f"✅ 성공: {result}")
    except Exception as e:
        print(f"❌ 최종 실패: {e}")
    print()

    print("--- 지수 백오프 재시도 테스트 ---")
    try:
        result = test_backoff_db()
        print(f"✅ 성공: {result}")
    except Exception as e:
        print(f"❌ 최종 실패: {e}")
    print()

    print("--- 서킷 브레이커 테스트 ---")
    print("서킷 브레이커 여러 번 호출하여 상태 변화 관찰:")
    for i in range(8):
        try:
            result = test_circuit_breaker()
            print(f"호출 {i+1}: ✅ 성공 - {result}")
        except Exception as e:
            print(f"호출 {i+1}: ❌ 실패 - {e}")
        time.sleep(0.5)
    print()

    print("--- 폴백 데코레이터 테스트 ---")
    for i in range(3):
        try:
            result = test_fallback()
            print(f"시도 {i+1}: ✅ 결과 - {result}")
        except Exception as e:
            print(f"시도 {i+1}: ❌ 오류 - {e}")
    print()

    print("--- 여러 데코레이터 조합 테스트 ---")
    try:
        result = test_combined_decorators()
        print(f"✅ 조합 데코레이터 결과: {result}")
    except Exception as e:
        print(f"❌ 조합 데코레이터 실패: {e}")
    print()

    print("--- 서킷 브레이커 복구 테스트 ---")
    print("잠시 기다린 후 서킷 브레이커 복구 상태 테스트...")
    time.sleep(2)  # recovery_timeout보다 짧게 설정했으므로 여전히 차단 상태
    try:
        result = test_circuit_breaker()
        print(f"✅ 복구 후 성공: {result}")
    except Exception as e:
        print(f"❌ 아직 차단 상태: {e}")

    print("\n=== 테스트 완료 ===")
    print("다양한 에러 처리 패턴을 학습했습니다!")
    print("- 기본 재시도")
    print("- 지수 백오프")
    print("- 서킷 브레이커")
    print("- 폴백 메커니즘")
    print("- 데코레이터 조합")