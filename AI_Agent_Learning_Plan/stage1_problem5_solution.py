# 1단계 문제 5: 에러 처리 데코레이터 - 해답

import time
import functools
import random
from datetime import datetime, timedelta
from typing import Any, Callable, Tuple, Type, Union, Optional, Dict
from enum import Enum

class CircuitBreakerState(Enum):
    """서킷 브레이커 상태"""
    CLOSED = "closed"      # 정상 상태
    OPEN = "open"          # 차단 상태
    HALF_OPEN = "half_open"  # 반열림 상태

class CircuitBreakerInfo:
    """서킷 브레이커 정보를 저장하는 클래스"""

    def __init__(self, failure_threshold: int, recovery_timeout: float):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = CircuitBreakerState.CLOSED

    def should_allow_request(self) -> bool:
        """요청을 허용할지 결정"""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            # 복구 시간이 지났는지 확인
            if (self.last_failure_time and
                datetime.now() - self.last_failure_time >= timedelta(seconds=self.recovery_timeout)):
                self.state = CircuitBreakerState.HALF_OPEN
                return True
            return False
        else:  # HALF_OPEN
            return True

    def on_success(self):
        """성공 시 호출"""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0

    def on_failure(self):
        """실패 시 호출"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.failure_count >= self.failure_threshold:
            if self.state != CircuitBreakerState.OPEN:
                self.state = CircuitBreakerState.OPEN

# 전역 서킷 브레이커 상태 저장
circuit_breaker_states: Dict[str, CircuitBreakerInfo] = {}

def retry_decorator(max_retries: int = 3, delay: float = 1.0,
                   exceptions: Tuple[Type[Exception], ...] = (Exception,)):
    """
    기본 재시도 데코레이터

    지정된 예외가 발생할 때 함수를 재시도합니다.

    Args:
        max_retries: 최대 재시도 횟수
        delay: 재시도 간격 (초)
        exceptions: 재시도할 예외 타입들

    Returns:
        데코레이터 함수
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):  # +1은 첫 시도 포함
                try:
                    if attempt > 0:
                        print(f"🔄 재시도 {attempt}/{max_retries} - {func.__name__}")
                        time.sleep(delay)

                    result = func(*args, **kwargs)

                    if attempt > 0:
                        print(f"✅ {func.__name__} 재시도 성공 (시도 {attempt + 1}/{max_retries + 1})")

                    return result

                except Exception as e:
                    last_exception = e

                    # 지정된 예외 타입인지 확인
                    if not isinstance(e, exceptions):
                        print(f"❌ {func.__name__} - 재시도하지 않는 예외: {type(e).__name__}: {e}")
                        raise

                    if attempt < max_retries:
                        print(f"⚠️  {func.__name__} 실패 (시도 {attempt + 1}/{max_retries + 1}): {e}")
                    else:
                        print(f"❌ {func.__name__} 최종 실패 (모든 재시도 소진): {e}")

            # 모든 재시도 실패
            raise last_exception

        return wrapper
    return decorator

def exponential_backoff_retry(max_retries: int = 3, initial_delay: float = 1.0,
                            backoff_factor: float = 2.0, max_delay: float = 60.0,
                            exceptions: Tuple[Type[Exception], ...] = (Exception,)):
    """
    지수 백오프 재시도 데코레이터

    재시도할 때마다 대기 시간이 지수적으로 증가합니다.

    Args:
        max_retries: 최대 재시도 횟수
        initial_delay: 초기 대기 시간 (초)
        backoff_factor: 백오프 계수
        max_delay: 최대 대기 시간 (초)
        exceptions: 재시도할 예외 타입들

    Returns:
        데코레이터 함수
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    if attempt > 0:
                        # 지수 백오프 계산
                        delay = min(initial_delay * (backoff_factor ** (attempt - 1)), max_delay)
                        print(f"🔄 지수 백오프 재시도 {attempt}/{max_retries} - {func.__name__} (대기: {delay:.2f}초)")
                        time.sleep(delay)

                    result = func(*args, **kwargs)

                    if attempt > 0:
                        print(f"✅ {func.__name__} 지수 백오프 재시도 성공!")

                    return result

                except Exception as e:
                    last_exception = e

                    if not isinstance(e, exceptions):
                        print(f"❌ {func.__name__} - 백오프 재시도하지 않는 예외: {type(e).__name__}: {e}")
                        raise

                    if attempt < max_retries:
                        next_delay = min(initial_delay * (backoff_factor ** attempt), max_delay)
                        print(f"⚠️  {func.__name__} 실패 (시도 {attempt + 1}/{max_retries + 1}): {e} (다음 대기: {next_delay:.2f}초)")
                    else:
                        print(f"❌ {func.__name__} 지수 백오프 최종 실패: {e}")

            raise last_exception

        return wrapper
    return decorator

def circuit_breaker(failure_threshold: int = 5, recovery_timeout: float = 60.0,
                   expected_exception: Type[Exception] = Exception):
    """
    서킷 브레이커 데코레이터

    연속된 실패가 임계값에 도달하면 함수 호출을 차단합니다.

    Args:
        failure_threshold: 실패 임계값
        recovery_timeout: 복구 대기 시간 (초)
        expected_exception: 감지할 예외 타입

    Returns:
        데코레이터 함수
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__

            # 함수별 서킷 브레이커 상태 초기화
            if func_name not in circuit_breaker_states:
                circuit_breaker_states[func_name] = CircuitBreakerInfo(
                    failure_threshold, recovery_timeout
                )

            cb_info = circuit_breaker_states[func_name]

            # 요청 허용 여부 확인
            if not cb_info.should_allow_request():
                remaining_time = recovery_timeout - (datetime.now() - cb_info.last_failure_time).total_seconds()
                raise RuntimeError(
                    f"🚫 서킷 브레이커 OPEN - {func_name} 호출 차단됨 "
                    f"(복구까지 {remaining_time:.1f}초 남음)"
                )

            # 상태 변화 로그
            if cb_info.state == CircuitBreakerState.HALF_OPEN:
                print(f"🔄 서킷 브레이커 HALF-OPEN - {func_name} 테스트 호출 허용")

            try:
                result = func(*args, **kwargs)

                # 성공 처리
                old_state = cb_info.state
                cb_info.on_success()

                if old_state != cb_info.state:
                    print(f"✅ 서킷 브레이커 {old_state.value} → {cb_info.state.value} - {func_name}")

                return result

            except Exception as e:
                # 예상된 예외인지 확인
                if isinstance(e, expected_exception):
                    old_state = cb_info.state
                    cb_info.on_failure()

                    print(f"⚠️  서킷 브레이커 실패 카운트: {cb_info.failure_count}/{failure_threshold} - {func_name}")

                    if old_state != cb_info.state:
                        print(f"🚫 서킷 브레이커 {old_state.value} → {cb_info.state.value} - {func_name} (복구: {recovery_timeout}초 후)")

                raise

        return wrapper
    return decorator

def fallback_decorator(fallback_func: Callable, default_value: Any = None):
    """
    폴백 데코레이터

    메인 함수 실행 실패 시 대체 함수를 호출하거나 기본값을 반환합니다.

    Args:
        fallback_func: 대체 함수
        default_value: 기본 반환값

    Returns:
        데코레이터 함수
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 메인 함수 시도
            try:
                result = func(*args, **kwargs)
                print(f"✅ {func.__name__} 메인 함수 성공")
                return result

            except Exception as main_error:
                print(f"⚠️  {func.__name__} 메인 함수 실패: {main_error}")

                # 폴백 함수 시도
                if fallback_func:
                    try:
                        result = fallback_func(*args, **kwargs)
                        print(f"✅ {func.__name__} 폴백 함수 성공")
                        return result

                    except Exception as fallback_error:
                        print(f"⚠️  {func.__name__} 폴백 함수도 실패: {fallback_error}")

                # 기본값 반환
                if default_value is not None:
                    print(f"🔄 {func.__name__} 기본값 반환: {default_value}")
                    return default_value

                # 모든 방법 실패
                print(f"❌ {func.__name__} 모든 방법 실패")
                raise main_error

        return wrapper
    return decorator

# 서킷 브레이커 상태 조회 함수
def get_circuit_breaker_status(func_name: str) -> Dict[str, Any]:
    """서킷 브레이커 상태 조회"""
    if func_name not in circuit_breaker_states:
        return {"error": "함수를 찾을 수 없습니다"}

    cb_info = circuit_breaker_states[func_name]
    return {
        "function": func_name,
        "state": cb_info.state.value,
        "failure_count": cb_info.failure_count,
        "failure_threshold": cb_info.failure_threshold,
        "last_failure_time": cb_info.last_failure_time.isoformat() if cb_info.last_failure_time else None,
        "recovery_timeout": cb_info.recovery_timeout
    }

def reset_circuit_breaker(func_name: str) -> bool:
    """서킷 브레이커 상태 초기화"""
    if func_name in circuit_breaker_states:
        cb_info = circuit_breaker_states[func_name]
        cb_info.state = CircuitBreakerState.CLOSED
        cb_info.failure_count = 0
        cb_info.last_failure_time = None
        print(f"🔄 서킷 브레이커 초기화됨: {func_name}")
        return True
    return False

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

# 복잡한 조합 테스트
@circuit_breaker(failure_threshold=2, recovery_timeout=5.0, expected_exception=ConnectionError)
@retry_decorator(max_retries=2, delay=0.1, exceptions=(ConnectionError,))
def test_complex_combination():
    return unreliable_api_call(success_rate=0.1)

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
            print(f"호출 {i+1}: ❌ 실패 - {str(e)[:50]}...")

        # 상태 확인
        status = get_circuit_breaker_status("test_circuit_breaker")
        print(f"  현재 상태: {status['state']}, 실패 횟수: {status['failure_count']}/{status['failure_threshold']}")

        time.sleep(0.2)
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

    print("--- 복잡한 조합 테스트 (서킷브레이커 + 재시도) ---")
    for i in range(5):
        try:
            result = test_complex_combination()
            print(f"복잡 조합 {i+1}: ✅ 성공 - {result}")
        except Exception as e:
            print(f"복잡 조합 {i+1}: ❌ 실패 - {str(e)[:40]}...")

        time.sleep(0.3)
    print()

    print("--- 서킷 브레이커 복구 테스트 ---")
    print("서킷 브레이커 상태 확인:")
    for func_name in ["test_circuit_breaker", "test_complex_combination"]:
        status = get_circuit_breaker_status(func_name)
        if "error" not in status:
            print(f"  {func_name}: {status['state']} (실패: {status['failure_count']}/{status['failure_threshold']})")

    print("\n서킷 브레이커 초기화 테스트:")
    reset_circuit_breaker("test_circuit_breaker")

    print("\n초기화 후 호출 테스트:")
    try:
        result = test_circuit_breaker()
        print(f"✅ 초기화 후 성공: {result}")
    except Exception as e:
        print(f"❌ 초기화 후에도 실패: {e}")

    print("\n=== 테스트 완료 ===")
    print("🎯 학습한 에러 처리 패턴:")
    print("  - 기본 재시도 (고정 간격)")
    print("  - 지수 백오프 (증가하는 간격)")
    print("  - 서킷 브레이커 (자동 차단/복구)")
    print("  - 폴백 메커니즘 (대체 실행)")
    print("  - 데코레이터 조합 (복합 전략)")
    print("  - 상태 모니터링 (실시간 추적)")

"""
학습 포인트:
1. 에러 처리 데코레이터의 다양한 패턴
2. 지수 백오프 알고리즘 구현
3. 서킷 브레이커 패턴과 상태 관리
4. 폴백 메커니즘을 통한 견고성 향상
5. 여러 데코레이터의 조합 사용법
6. 예외 타입별 다른 처리 전략
7. 시간 기반 로직과 상태 추적
8. 실무에서 사용되는 신뢰성 패턴들

이러한 에러 처리 패턴들은 실제 AI Agent 개발에서 매우 중요합니다.
특히 외부 API 호출, 데이터베이스 연결, 네트워크 통신에서
안정성과 신뢰성을 확보하는 데 필수적입니다.
"""