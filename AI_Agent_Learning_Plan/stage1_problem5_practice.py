# 1단계 문제 5 연습: 에러 처리 데코레이터
# API 호출 실패 시 재시도하는 데코레이터를 구현해보세요.

"""
학습 목표:
- 에러 처리 패턴 학습
- 재시도 로직 구현
- 지수 백오프 알고리즘 이해
- 서킷 브레이커 패턴 구현
- 폴백 메커니즘 구현

구현할 기능:
1. retry_decorator: 기본 재시도 데코레이터
2. exponential_backoff_retry: 지수 백오프 재시도
3. circuit_breaker: 서킷 브레이커 패턴
4. fallback_decorator: 폴백 메커니즘

핵심 개념:
- 예외 처리: try/except를 통한 에러 캐치
- 재시도 로직: 실패 시 설정된 횟수만큼 재시도
- 지수 백오프: 재시도할 때마다 대기 시간을 지수적으로 증가
- 서킷 브레이커: 연속 실패 시 호출을 차단하여 시스템 보호
- 폴백: 주 기능 실패 시 대체 기능으로 전환

힌트:
- isinstance(exception, exception_types)로 예외 타입 확인
- time.sleep()으로 재시도 간격 구현
- 전역 딕셔너리로 서킷 브레이커 상태 관리
- datetime으로 시간 기반 로직 구현
"""

import time
import functools
import random
from datetime import datetime, timedelta
from typing import Any, Callable, Tuple, Type, Union, Optional, Dict

class CircuitBreakerState:
    """서킷 브레이커 상태 관리 클래스"""

    def __init__(self, failure_threshold: int, recovery_timeout: float):
        """
        서킷 브레이커 상태 초기화

        Args:
            failure_threshold: 연속 실패 임계값
            recovery_timeout: 복구 대기 시간 (초)
        """
        # TODO: 필요한 속성들 초기화
        # self.failure_count = 0  # 현재 연속 실패 횟수
        # self.failure_threshold = failure_threshold  # 실패 임계값
        # self.recovery_timeout = recovery_timeout  # 복구 대기 시간
        # self.last_failure_time = None  # 마지막 실패 시간
        # self.state = "closed"  # 상태: "closed", "open", "half_open"
        pass

    def record_success(self):
        """성공 시 호출 - 실패 카운터 리셋"""
        # TODO: 실패 횟수를 0으로 리셋하고 상태를 "closed"로 변경
        pass

    def record_failure(self):
        """실패 시 호출 - 실패 카운터 증가"""
        # TODO: 실패 횟수 증가
        # TODO: 임계값 도달 시 상태를 "open"으로 변경하고 실패 시간 기록
        pass

    def can_execute(self) -> bool:
        """함수 실행 가능 여부 확인"""
        if self.state == "closed":
            return True
        elif self.state == "open":
            # TODO: 복구 대기 시간이 지났는지 확인
            # if datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout):
            #     self.state = "half_open"
            #     return True
            # return False
            pass
        else:  # half_open
            return True

# 전역 서킷 브레이커 상태 저장
circuit_breaker_states: Dict[str, CircuitBreakerState] = {}

def retry_decorator(max_retries: int = 3, delay: float = 1.0,
                   exceptions: Tuple[Type[Exception], ...] = (Exception,)):
    """
    기본 재시도 데코레이터

    Args:
        max_retries: 최대 재시도 횟수
        delay: 재시도 간격 (초)
        exceptions: 재시도할 예외 타입들
    """
    def decorator(func: Callable) -> Callable:
        # TODO: @functools.wraps(func) 사용
        def wrapper(*args, **kwargs):
            last_exception = None

            # TODO: max_retries + 1번 시도 (첫 실행 + 재시도)
            for attempt in range(max_retries + 1):
                try:
                    # TODO: 함수 실행 및 성공 시 결과 반환
                    pass
                except Exception as e:
                    # TODO: 지정된 예외 타입인지 확인
                    if isinstance(e, exceptions):
                        last_exception = e
                        if attempt < max_retries:
                            # TODO: 재시도 로그 출력
                            # print(f"[재시도 {attempt + 1}/{max_retries}] {func.__name__} 실패: {e}")
                            # TODO: 지연 시간만큼 대기
                            pass
                        else:
                            # TODO: 최대 재시도 횟수 도달 시 로그
                            # print(f"[재시도 실패] {func.__name__} 최대 재시도 횟수 초과")
                            pass
                    else:
                        # TODO: 재시도하지 않는 예외인 경우 바로 발생
                        pass

            # TODO: 모든 재시도 실패 시 마지막 예외 발생
            pass

        # TODO: wrapper 반환
        pass

    # TODO: decorator 반환
    pass

def exponential_backoff_retry(max_retries: int = 3, initial_delay: float = 1.0,
                            backoff_factor: float = 2.0, max_delay: float = 60.0,
                            exceptions: Tuple[Type[Exception], ...] = (Exception,)):
    """
    지수 백오프 재시도 데코레이터

    Args:
        max_retries: 최대 재시도 횟수
        initial_delay: 초기 지연 시간
        backoff_factor: 백오프 계수
        max_delay: 최대 지연 시간
        exceptions: 재시도할 예외 타입들
    """
    def decorator(func: Callable) -> Callable:
        # TODO: @functools.wraps(func) 사용
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = initial_delay

            for attempt in range(max_retries + 1):
                try:
                    # TODO: 함수 실행 및 성공 시 결과 반환
                    pass
                except Exception as e:
                    if isinstance(e, exceptions):
                        last_exception = e
                        if attempt < max_retries:
                            # TODO: 지수 백오프 로그 출력
                            # print(f"[지수 백오프 재시도 {attempt + 1}/{max_retries}] {func.__name__} 실패: {e}")
                            # print(f"[대기 시간] {current_delay:.2f}초 대기 중...")

                            # TODO: 현재 지연 시간만큼 대기

                            # TODO: 다음 재시도를 위해 지연 시간 증가 (최대값 제한)
                            # current_delay = min(current_delay * backoff_factor, max_delay)
                            pass
                        else:
                            # TODO: 최대 재시도 횟수 도달 로그
                            pass
                    else:
                        # TODO: 재시도하지 않는 예외 발생
                        pass

            # TODO: 모든 재시도 실패 시 마지막 예외 발생
            pass

        # TODO: wrapper 반환
        pass

    # TODO: decorator 반환
    pass

def circuit_breaker(failure_threshold: int = 5, recovery_timeout: float = 60.0,
                   expected_exception: Type[Exception] = Exception):
    """
    서킷 브레이커 데코레이터

    Args:
        failure_threshold: 연속 실패 임계값
        recovery_timeout: 복구 대기 시간
        expected_exception: 감지할 예외 타입
    """
    def decorator(func: Callable) -> Callable:
        # TODO: @functools.wraps(func) 사용
        def wrapper(*args, **kwargs):
            func_name = func.__name__

            # TODO: 함수별 서킷 브레이커 상태 생성 또는 가져오기
            if func_name not in circuit_breaker_states:
                circuit_breaker_states[func_name] = CircuitBreakerState(
                    failure_threshold, recovery_timeout
                )

            state = circuit_breaker_states[func_name]

            # TODO: 실행 가능 여부 확인
            if not state.can_execute():
                # TODO: 차단 상태 로그 및 예외 발생
                # print(f"[서킷 브레이커 차단] {func_name} 호출이 차단되었습니다.")
                # raise RuntimeError("서킷 브레이커가 열린 상태입니다.")
                pass

            try:
                # TODO: 함수 실행
                # result = func(*args, **kwargs)
                # TODO: 성공 시 상태 업데이트
                # state.record_success()
                # if state.state == "half_open":
                #     print(f"[서킷 브레이커 복구] {func_name} 정상 동작을 확인했습니다.")
                # return result
                pass
            except Exception as e:
                if isinstance(e, expected_exception):
                    # TODO: 실패 기록
                    # state.record_failure()
                    # if state.state == "open":
                    #     print(f"[서킷 브레이커 활성화] {func_name} 연속 실패로 인해 차단됩니다.")
                    pass
                # TODO: 예외 재발생
                pass

        # TODO: wrapper 반환
        pass

    # TODO: decorator 반환
    pass

def fallback_decorator(fallback_func: Callable, default_value: Any = None):
    """
    폴백 데코레이터

    Args:
        fallback_func: 대체 함수
        default_value: 대체 함수도 실패할 경우의 기본값
    """
    def decorator(func: Callable) -> Callable:
        # TODO: @functools.wraps(func) 사용
        def wrapper(*args, **kwargs):
            try:
                # TODO: 주 함수 실행
                # result = func(*args, **kwargs)
                # print(f"[주 함수 성공] {func.__name__}")
                # return result
                pass
            except Exception as e:
                # TODO: 주 함수 실패 로그
                # print(f"[주 함수 실패] {func.__name__}: {e}")

                try:
                    # TODO: 폴백 함수 실행
                    # result = fallback_func()
                    # print(f"[폴백 성공] 대체 함수로 결과 획득")
                    # return result
                    pass
                except Exception as fallback_error:
                    # TODO: 폴백 함수도 실패한 경우
                    # print(f"[폴백 실패] 대체 함수도 실패: {fallback_error}")
                    # print(f"[기본값 사용] 기본값 반환: {default_value}")
                    # return default_value
                    pass

        # TODO: wrapper 반환
        pass

    # TODO: decorator 반환
    pass

# 테스트용 함수들
def unreliable_api_call(success_rate: float = 0.3, response_time: float = 0.1):
    """불안정한 API 호출을 시뮬레이션하는 함수"""
    time.sleep(response_time)
    if random.random() < success_rate:
        return f"API 호출 성공! 시간: {datetime.now().strftime('%H:%M:%S')}"
    else:
        raise ConnectionError("API 서버 연결 실패")

def database_query(success_rate: float = 0.4):
    """데이터베이스 쿼리 시뮬레이션"""
    time.sleep(0.05)
    if random.random() < success_rate:
        return "데이터베이스 쿼리 결과: [1, 2, 3, 4, 5]"
    else:
        raise TimeoutError("데이터베이스 연결 타임아웃")

def external_service_call():
    """외부 서비스 호출 시뮬레이션"""
    time.sleep(0.02)
    if random.random() < 0.2:  # 20% 성공률
        return "외부 서비스 응답 데이터"
    else:
        raise RuntimeError("외부 서비스 오류")

# 폴백 함수들
def api_fallback():
    """API 호출 실패 시 대체 함수"""
    time.sleep(0.02)
    if random.random() < 0.8:  # 80% 성공률
        return "캐시된 API 응답"
    else:
        raise Exception("폴백도 실패")

def db_fallback():
    """데이터베이스 쿼리 실패 시 대체 함수"""
    return "기본 데이터: [0, 0, 0]"

# 데코레이터가 적용된 테스트 함수들
@retry_decorator(max_retries=3, delay=0.1, exceptions=(ConnectionError,))
def test_retry_api():
    return unreliable_api_call(success_rate=0.4)

@exponential_backoff_retry(max_retries=4, initial_delay=0.05, backoff_factor=2.0, max_delay=1.0)
def test_backoff_db():
    return database_query(success_rate=0.3)

@circuit_breaker(failure_threshold=3, recovery_timeout=2.0, expected_exception=RuntimeError)
def test_circuit_breaker():
    return external_service_call()

@fallback_decorator(fallback_func=api_fallback, default_value="기본 응답")
def test_fallback():
    return unreliable_api_call(success_rate=0.1)

@fallback_decorator(fallback_func=db_fallback, default_value="최종 기본값")
@retry_decorator(max_retries=2, delay=0.05, exceptions=(TimeoutError,))
def test_combined_decorators():
    return database_query(success_rate=0.2)

# 테스트 코드 - 완성한 후에 실행해보세요
if __name__ == "__main__":
    print("=== 에러 처리 데코레이터 연습 ===\n")

    # random 시드 설정 (일관된 테스트를 위해)
    random.seed(42)

    print("1. 기본 재시도 데코레이터 테스트")
    try:
        result = test_retry_api()
        print(f"✅ 성공: {result}")
    except Exception as e:
        print(f"❌ 최종 실패: {e}")
    print()

    print("2. 지수 백오프 재시도 테스트")
    try:
        result = test_backoff_db()
        print(f"✅ 성공: {result}")
    except Exception as e:
        print(f"❌ 최종 실패: {e}")
    print()

    print("3. 서킷 브레이커 테스트")
    print("여러 번 호출하여 서킷 브레이커 상태 변화 관찰:")
    for i in range(6):
        try:
            result = test_circuit_breaker()
            print(f"호출 {i+1}: ✅ 성공 - {result}")
        except Exception as e:
            print(f"호출 {i+1}: ❌ 실패 - {e}")
        time.sleep(0.1)
    print()

    print("4. 폴백 데코레이터 테스트")
    for i in range(3):
        try:
            result = test_fallback()
            print(f"시도 {i+1}: ✅ 결과 - {result}")
        except Exception as e:
            print(f"시도 {i+1}: ❌ 오류 - {e}")
    print()

    print("5. 여러 데코레이터 조합 테스트")
    try:
        result = test_combined_decorators()
        print(f"✅ 조합 데코레이터 결과: {result}")
    except Exception as e:
        print(f"❌ 조합 데코레이터 실패: {e}")
    print()

    print("6. 서킷 브레이커 복구 테스트")
    print("잠시 기다린 후 서킷 브레이커 복구 상태 테스트...")
    time.sleep(1)  # recovery_timeout보다 짧게 설정
    try:
        result = test_circuit_breaker()
        print(f"✅ 복구 후 성공: {result}")
    except Exception as e:
        print(f"❌ 아직 차단 상태: {e}")

    print("\n=== 연습 완료 ===")
    print("구현이 완료되면 다음 에러 처리 패턴을 학습하게 됩니다:")
    print("- 기본 재시도 로직")
    print("- 지수 백오프 알고리즘")
    print("- 서킷 브레이커 패턴")
    print("- 폴백 메커니즘")
    print("- 여러 데코레이터 조합")