# 1단계 문제 4: 실행 시간 측정 데코레이터
# Agent의 메서드 실행 시간을 측정하는 데코레이터를 만들어보세요.

"""
문제 요구사항:
1. timing_decorator 함수를 만드세요:
   - 함수의 실행 시간을 측정하고 출력
   - 함수 실행 전후에 로그 메시지 출력
   - 실행 시간을 초 단위로 소수점 4자리까지 표시

2. 고급 버전 advanced_timing_decorator를 만드세요:
   - 실행 시간 임계값을 설정 가능 (기본값: 1초)
   - 임계값을 초과하면 경고 메시지 출력
   - 함수 이름과 인수 정보도 함께 로그에 출력
   - 실행 결과도 로그에 기록

3. 클래스 메서드용 method_timer 데코레이터를 만드세요:
   - 클래스 메서드에 특화된 타이밍 측정
   - 클래스 이름과 메서드 이름을 함께 출력
   - self 매개변수는 로그에서 제외

4. 성능 통계를 수집하는 performance_tracker 데코레이터를 만드세요:
   - 함수별 호출 횟수, 총 실행 시간, 평균 실행 시간 추적
   - get_performance_stats() 함수로 통계 조회 가능
   - 통계 초기화 기능

힌트:
- time.time() 또는 time.perf_counter()를 사용하여 시간 측정
- functools.wraps를 사용하여 원본 함수의 메타데이터 보존
- *args, **kwargs를 사용하여 다양한 함수 서명 지원
- 클로저를 활용하여 설정 값을 저장
- 전역 딕셔너리를 사용하여 성능 통계 저장
"""

import time
import functools
from typing import Any, Callable, Dict, Optional

# 성능 통계 저장용 전역 딕셔너리
performance_stats: Dict[str, Dict[str, Any]] = {}

def timing_decorator(func: Callable) -> Callable:
    """
    기본 실행 시간 측정 데코레이터
    여기에 코드를 작성하세요.
    """
    pass

def advanced_timing_decorator(threshold: float = 1.0, show_args: bool = True, show_result: bool = False):
    """
    고급 실행 시간 측정 데코레이터
    여기에 코드를 작성하세요.
    """
    pass

def method_timer(func: Callable) -> Callable:
    """
    클래스 메서드용 타이밍 측정 데코레이터
    여기에 코드를 작성하세요.
    """
    pass

def performance_tracker(func: Callable) -> Callable:
    """
    성능 통계를 수집하는 데코레이터
    여기에 코드를 작성하세요.
    """
    pass

def get_performance_stats() -> Dict[str, Dict[str, Any]]:
    """
    수집된 성능 통계를 반환합니다.
    여기에 코드를 작성하세요.
    """
    pass

def clear_performance_stats() -> None:
    """
    성능 통계를 초기화합니다.
    여기에 코드를 작성하세요.
    """
    pass

# 테스트용 Agent 클래스
class TestAgent:
    def __init__(self, name: str):
        self.name = name

    @timing_decorator
    def quick_task(self):
        """빠른 작업 (0.1초)"""
        time.sleep(0.1)
        return "Quick task completed"

    @advanced_timing_decorator(threshold=0.5, show_args=True, show_result=True)
    def medium_task(self, task_name: str):
        """중간 작업 (0.7초)"""
        time.sleep(0.7)
        return f"Medium task '{task_name}' completed"

    @method_timer
    def slow_task(self, duration: float = 1.2):
        """느린 작업 (사용자 정의 시간)"""
        time.sleep(duration)
        return f"Slow task completed in {duration} seconds"

    @performance_tracker
    def repeated_task(self, data: str):
        """반복 실행되는 작업"""
        time.sleep(0.2)
        return f"Processed: {data}"

# 테스트 함수들
@timing_decorator
def simple_function():
    """간단한 함수 테스트"""
    time.sleep(0.3)
    return "Simple function result"

@advanced_timing_decorator(threshold=0.2, show_args=True, show_result=True)
def complex_function(x: int, y: int, operation: str = "add"):
    """복잡한 함수 테스트"""
    time.sleep(0.5)
    if operation == "add":
        result = x + y
    elif operation == "multiply":
        result = x * y
    else:
        result = 0
    return result

@performance_tracker
def data_processing_function(data_size: int):
    """데이터 처리 함수 (성능 추적용)"""
    # 데이터 크기에 비례한 처리 시간 시뮬레이션
    time.sleep(data_size * 0.1)
    return f"Processed {data_size} items"

# 테스트 코드
if __name__ == "__main__":
    print("=== 1단계 문제 4: 실행 시간 측정 데코레이터 테스트 ===\n")

    # 기본 타이밍 데코레이터 테스트
    print("--- 기본 타이밍 데코레이터 테스트 ---")
    result = simple_function()
    print(f"결과: {result}\n")

    # 고급 타이밍 데코레이터 테스트
    print("--- 고급 타이밍 데코레이터 테스트 ---")
    result = complex_function(10, 20, "multiply")
    print(f"결과: {result}\n")

    # 클래스 메서드 테스트
    print("--- 클래스 메서드 타이밍 테스트 ---")
    agent = TestAgent("TestBot")

    agent.quick_task()
    print()

    agent.medium_task("데이터 분석")
    print()

    agent.slow_task(0.8)
    print()

    # 성능 추적 데코레이터 테스트
    print("--- 성능 추적 데코레이터 테스트 ---")

    # 여러 번 실행하여 통계 수집
    for i in range(3):
        agent.repeated_task(f"data_{i}")

    for size in [1, 2, 3]:
        data_processing_function(size)

    # 동일 함수를 다시 실행
    data_processing_function(1)
    agent.repeated_task("final_data")

    print()

    # 성능 통계 확인
    print("--- 성능 통계 확인 ---")
    stats = get_performance_stats()
    for func_name, stat in stats.items():
        print(f"함수: {func_name}")
        print(f"  호출 횟수: {stat['call_count']}")
        print(f"  총 실행 시간: {stat['total_time']:.4f}초")
        print(f"  평균 실행 시간: {stat['average_time']:.4f}초")
        print(f"  최소 실행 시간: {stat['min_time']:.4f}초")
        print(f"  최대 실행 시간: {stat['max_time']:.4f}초")
        print()

    # 통계 초기화 테스트
    print("--- 통계 초기화 테스트 ---")
    clear_performance_stats()
    print("통계가 초기화되었습니다.")

    # 초기화 후 함수 실행
    data_processing_function(1)

    # 초기화된 통계 확인
    stats = get_performance_stats()
    print(f"초기화 후 통계: {len(stats)}개 함수")
    for func_name, stat in stats.items():
        print(f"  {func_name}: {stat['call_count']}회 호출")