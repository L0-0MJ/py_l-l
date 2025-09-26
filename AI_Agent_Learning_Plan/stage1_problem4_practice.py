# 1단계 문제 4 연습: 실행 시간 측정 데코레이터
# Agent의 메서드 실행 시간을 측정하는 데코레이터를 구현해보세요.

"""
학습 목표:
- 데코레이터 함수 작성법 이해
- functools.wraps 사용법 학습
- 클로저(closure) 개념 이해
- 시간 측정 및 로깅 구현
- 전역 상태 관리

구현할 기능:
1. timing_decorator: 기본 실행 시간 측정
2. advanced_timing_decorator: 고급 기능이 포함된 시간 측정
3. method_timer: 클래스 메서드 전용 타이머
4. performance_tracker: 성능 통계 수집

핵심 개념:
- 데코레이터는 함수를 인자로 받아 새로운 함수를 반환
- functools.wraps는 원본 함수의 메타데이터를 보존
- *args, **kwargs로 다양한 함수 시그니처 지원
- time.perf_counter()로 정확한 시간 측정

힌트:
- 데코레이터 구조: def decorator(func): def wrapper(*args, **kwargs): ... return wrapper
- 시간 측정: start = time.perf_counter(); result = func(*args, **kwargs); end = time.perf_counter()
- 파라미터가 있는 데코레이터: def decorator(param): def real_decorator(func): def wrapper(...): ...
"""

import time
import functools
from typing import Any, Callable, Dict, Optional

# 성능 통계 저장용 전역 딕셔너리
performance_stats: Dict[str, Dict[str, Any]] = {}

def timing_decorator(func: Callable) -> Callable:
    """
    기본 실행 시간 측정 데코레이터

    기능:
    - 함수 실행 전후에 로그 메시지 출력
    - 실행 시간을 초 단위로 소수점 4자리까지 표시
    """
    # TODO: @functools.wraps(func)를 사용해서 원본 함수 메타데이터 보존
    def wrapper(*args, **kwargs):
        # TODO: 실행 시작 로그 출력
        # print(f"[실행 시작] {func.__name__}")

        # TODO: 시작 시간 기록 (time.perf_counter() 사용)

        # TODO: 원본 함수 실행

        # TODO: 종료 시간 기록

        # TODO: 실행 시간 계산 및 로그 출력
        # print(f"[실행 완료] {func.__name__} - 실행시간: {실행시간:.4f}초")

        # TODO: 함수 결과 반환
        pass

    # TODO: wrapper 함수 반환
    pass

def advanced_timing_decorator(threshold: float = 1.0, show_args: bool = True, show_result: bool = False):
    """
    고급 실행 시간 측정 데코레이터 (파라미터 있는 데코레이터)

    Args:
        threshold: 경고 임계값 (초)
        show_args: 함수 인수 표시 여부
        show_result: 함수 결과 표시 여부
    """
    def decorator(func: Callable) -> Callable:
        # TODO: @functools.wraps(func) 사용
        def wrapper(*args, **kwargs):
            # TODO: 함수 이름과 인수 정보 준비
            func_info = f"{func.__name__}"
            if show_args and (args or kwargs):
                args_str = ", ".join([str(arg) for arg in args])
                kwargs_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
                all_args = ", ".join(filter(None, [args_str, kwargs_str]))
                func_info += f"({all_args})"

            # TODO: 실행 시작 로그
            # print(f"[고급 타이머 시작] {func_info}")

            # TODO: 시간 측정 및 함수 실행

            # TODO: 실행 시간 계산

            # TODO: 임계값 초과 시 경고 메시지
            # if 실행시간 > threshold:
            #     print(f"⚠️  [성능 경고] {func.__name__} 실행시간이 임계값({threshold}초)을 초과했습니다!")

            # TODO: 결과 로깅 (show_result가 True인 경우)

            # TODO: 완료 로그 및 결과 반환
            pass

        # TODO: wrapper 반환
        pass

    # TODO: decorator 반환
    pass

def method_timer(func: Callable) -> Callable:
    """
    클래스 메서드용 타이밍 측정 데코레이터

    기능:
    - 클래스 이름과 메서드 이름을 함께 출력
    - self 매개변수는 로그에서 제외
    """
    # TODO: @functools.wraps(func) 사용
    def wrapper(*args, **kwargs):
        # TODO: self 객체에서 클래스 이름 추출
        if args:
            class_name = args[0].__class__.__name__
            method_info = f"{class_name}.{func.__name__}"
        else:
            method_info = func.__name__

        # TODO: 실행 시작 로그 (self 제외한 인수들 표시)
        # remaining_args = args[1:] if args else []

        # TODO: 시간 측정 및 실행

        # TODO: 완료 로그 및 결과 반환
        pass

    # TODO: wrapper 반환
    pass

def performance_tracker(func: Callable) -> Callable:
    """
    성능 통계를 수집하는 데코레이터

    수집 정보:
    - 호출 횟수
    - 총 실행 시간
    - 평균 실행 시간
    - 최소/최대 실행 시간
    """
    # TODO: @functools.wraps(func) 사용
    def wrapper(*args, **kwargs):
        # TODO: 함수 이름으로 통계 키 생성
        func_name = func.__name__

        # TODO: 시간 측정 및 함수 실행

        # TODO: 통계 업데이트
        if func_name not in performance_stats:
            performance_stats[func_name] = {
                'call_count': 0,
                'total_time': 0.0,
                'min_time': float('inf'),
                'max_time': 0.0,
                'average_time': 0.0
            }

        stats = performance_stats[func_name]
        # TODO: 통계 값들 업데이트
        # stats['call_count'] += 1
        # stats['total_time'] += 실행시간
        # stats['min_time'] = min(stats['min_time'], 실행시간)
        # stats['max_time'] = max(stats['max_time'], 실행시간)
        # stats['average_time'] = stats['total_time'] / stats['call_count']

        # TODO: 결과 반환
        pass

    # TODO: wrapper 반환
    pass

def get_performance_stats() -> Dict[str, Dict[str, Any]]:
    """수집된 성능 통계를 반환합니다."""
    # TODO: performance_stats 딕셔너리 반환
    pass

def clear_performance_stats() -> None:
    """성능 통계를 초기화합니다."""
    # TODO: performance_stats 딕셔너리 초기화
    pass

# 테스트용 클래스
class TestAgent:
    """데코레이터 테스트용 Agent 클래스"""

    def __init__(self, name: str):
        self.name = name

    @timing_decorator
    def quick_task(self):
        """빠른 작업 (0.1초)"""
        time.sleep(0.1)
        return "Quick task completed"

    @advanced_timing_decorator(threshold=0.05, show_args=True, show_result=True)
    def medium_task(self, task_name: str):
        """중간 작업 (0.3초)"""
        time.sleep(0.3)
        return f"Medium task '{task_name}' completed"

    @method_timer
    def slow_task(self, duration: float = 0.2):
        """느린 작업 (사용자 정의 시간)"""
        time.sleep(duration)
        return f"Slow task completed in {duration} seconds"

    @performance_tracker
    def repeated_task(self, data: str):
        """반복 실행되는 작업"""
        time.sleep(0.1)
        return f"Processed: {data}"

# 테스트 함수들
@timing_decorator
def simple_function():
    """간단한 함수 테스트"""
    time.sleep(0.05)
    return "Simple function result"

@advanced_timing_decorator(threshold=0.02, show_args=True, show_result=True)
def complex_function(x: int, y: int, operation: str = "add"):
    """복잡한 함수 테스트"""
    time.sleep(0.1)
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
    time.sleep(data_size * 0.02)
    return f"Processed {data_size} items"

# 테스트 코드 - 완성한 후에 실행해보세요
if __name__ == "__main__":
    print("=== 실행 시간 측정 데코레이터 연습 ===\n")

    # 1. 기본 타이밍 데코레이터 테스트
    print("1. 기본 타이밍 데코레이터")
    result = simple_function()
    print(f"결과: {result}\n")

    # 2. 고급 타이밍 데코레이터 테스트
    print("2. 고급 타이밍 데코레이터")
    result = complex_function(10, 20, "multiply")
    print(f"최종 결과: {result}\n")

    # 3. 클래스 메서드 타이밍 테스트
    print("3. 클래스 메서드 타이밍")
    agent = TestAgent("TestBot")

    agent.quick_task()
    print()

    agent.medium_task("데이터 분석")
    print()

    agent.slow_task(0.15)
    print()

    # 4. 성능 추적 데코레이터 테스트
    print("4. 성능 추적 데코레이터")

    # 여러 번 실행하여 통계 수집
    for i in range(3):
        agent.repeated_task(f"data_{i}")

    for size in [1, 2, 3]:
        data_processing_function(size)

    # 동일 함수를 다시 실행
    data_processing_function(1)
    agent.repeated_task("final_data")
    print()

    # 5. 성능 통계 확인
    print("5. 성능 통계 확인")
    stats = get_performance_stats()
    if stats:
        for func_name, stat in stats.items():
            print(f"함수: {func_name}")
            print(f"  호출 횟수: {stat['call_count']}")
            print(f"  총 실행 시간: {stat['total_time']:.4f}초")
            print(f"  평균 실행 시간: {stat['average_time']:.4f}초")
            print(f"  최소 실행 시간: {stat['min_time']:.4f}초")
            print(f"  최대 실행 시간: {stat['max_time']:.4f}초")
            print()
    else:
        print("통계 데이터가 없습니다. performance_tracker 구현을 확인하세요.")

    # 6. 통계 초기화 테스트
    print("6. 통계 초기화 테스트")
    clear_performance_stats()
    print("통계가 초기화되었습니다.")

    # 초기화 후 함수 실행
    data_processing_function(1)

    # 초기화된 통계 확인
    stats = get_performance_stats()
    print(f"초기화 후 통계: {len(stats)}개 함수")
    for func_name, stat in stats.items():
        print(f"  {func_name}: {stat['call_count']}회 호출")

    print("\n=== 연습 완료 ===")
    print("구현이 완료되면 다음을 학습하게 됩니다:")
    print("- 데코레이터 기본 구조")
    print("- functools.wraps 사용법")
    print("- 파라미터가 있는 데코레이터")
    print("- 성능 통계 수집")