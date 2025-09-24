# 1단계 문제 4: 실행 시간 측정 데코레이터 - 해답

import time
import functools
from typing import Any, Callable, Dict, Optional
from datetime import datetime

# 성능 통계 저장용 전역 딕셔너리
performance_stats: Dict[str, Dict[str, Any]] = {}

def timing_decorator(func: Callable) -> Callable:
    """
    기본 실행 시간 측정 데코레이터

    함수의 실행 시간을 측정하고 로그를 출력하는 간단한 데코레이터입니다.

    Args:
        func (Callable): 측정할 함수

    Returns:
        Callable: 래핑된 함수
    """
    @functools.wraps(func)  # 원본 함수의 메타데이터 보존
    def wrapper(*args, **kwargs):
        # 실행 전 로그
        print(f"⏱️  Starting '{func.__name__}' execution...")

        # 시작 시간 기록 (perf_counter는 더 정확한 시간 측정)
        start_time = time.perf_counter()

        try:
            # 원본 함수 실행
            result = func(*args, **kwargs)

            # 종료 시간 기록
            end_time = time.perf_counter()
            execution_time = end_time - start_time

            # 실행 후 로그
            print(f"✅ '{func.__name__}' completed in {execution_time:.4f} seconds")

            return result

        except Exception as e:
            # 에러 발생 시에도 시간 측정
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            print(f"❌ '{func.__name__}' failed after {execution_time:.4f} seconds: {str(e)}")
            raise

    return wrapper

def advanced_timing_decorator(threshold: float = 1.0, show_args: bool = True, show_result: bool = False):
    """
    고급 실행 시간 측정 데코레이터

    더 자세한 정보와 설정 가능한 옵션을 제공하는 데코레이터입니다.

    Args:
        threshold (float): 경고 출력 임계값 (초)
        show_args (bool): 함수 인수 표시 여부
        show_result (bool): 함수 결과 표시 여부

    Returns:
        Callable: 데코레이터 함수
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 함수 정보 수집
            func_name = func.__name__
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 인수 정보 생성
            args_info = ""
            if show_args:
                args_str = ", ".join(repr(arg) for arg in args)
                kwargs_str = ", ".join(f"{k}={repr(v)}" for k, v in kwargs.items())
                all_args = [s for s in [args_str, kwargs_str] if s]
                args_info = f" with args: ({', '.join(all_args)})" if all_args else ""

            # 실행 전 로그
            print(f"🚀 [{timestamp}] Starting '{func_name}'{args_info}")

            start_time = time.perf_counter()

            try:
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                execution_time = end_time - start_time

                # 결과 정보
                result_info = ""
                if show_result:
                    result_str = repr(result)
                    # 결과가 너무 길면 줄임
                    if len(result_str) > 100:
                        result_str = result_str[:97] + "..."
                    result_info = f" -> {result_str}"

                # 성능 경고 확인
                if execution_time > threshold:
                    print(f"⚠️  [{timestamp}] '{func_name}' SLOW execution: {execution_time:.4f}s (threshold: {threshold}s){result_info}")
                else:
                    print(f"✅ [{timestamp}] '{func_name}' completed: {execution_time:.4f}s{result_info}")

                return result

            except Exception as e:
                end_time = time.perf_counter()
                execution_time = end_time - start_time
                print(f"❌ [{timestamp}] '{func_name}' failed after {execution_time:.4f}s: {str(e)}")
                raise

        return wrapper
    return decorator

def method_timer(func: Callable) -> Callable:
    """
    클래스 메서드용 타이밍 측정 데코레이터

    클래스의 메서드에 특화된 시간 측정 데코레이터입니다.
    클래스 이름과 메서드 이름을 함께 표시합니다.

    Args:
        func (Callable): 측정할 메서드

    Returns:
        Callable: 래핑된 메서드
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # 클래스 이름 추출
        class_name = self.__class__.__name__
        method_name = func.__name__
        full_name = f"{class_name}.{method_name}"

        # 인스턴스 정보 (self는 제외하고 다른 인수만 표시)
        args_info = ""
        if args or kwargs:
            args_str = ", ".join(repr(arg) for arg in args)
            kwargs_str = ", ".join(f"{k}={repr(v)}" for k, v in kwargs.items())
            all_args = [s for s in [args_str, kwargs_str] if s]
            args_info = f"({', '.join(all_args)})" if all_args else "()"

        print(f"🔧 Method '{full_name}' started{args_info}")

        start_time = time.perf_counter()

        try:
            result = func(self, *args, **kwargs)
            end_time = time.perf_counter()
            execution_time = end_time - start_time

            print(f"🔧 Method '{full_name}' completed in {execution_time:.4f}s")
            return result

        except Exception as e:
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            print(f"🔧 Method '{full_name}' failed after {execution_time:.4f}s: {str(e)}")
            raise

    return wrapper

def performance_tracker(func: Callable) -> Callable:
    """
    성능 통계를 수집하는 데코레이터

    함수의 호출 횟수, 실행 시간 등을 추적하여 성능 분석을 위한
    통계 정보를 수집합니다.

    Args:
        func (Callable): 추적할 함수

    Returns:
        Callable: 래핑된 함수
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__

        # 함수별 통계 초기화
        if func_name not in performance_stats:
            performance_stats[func_name] = {
                'call_count': 0,
                'total_time': 0.0,
                'min_time': float('inf'),
                'max_time': 0.0,
                'average_time': 0.0,
                'last_called': None,
                'first_called': None
            }

        stats = performance_stats[func_name]

        # 호출 시간 기록
        call_time = datetime.now().isoformat()
        if stats['first_called'] is None:
            stats['first_called'] = call_time
        stats['last_called'] = call_time

        start_time = time.perf_counter()

        try:
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            execution_time = end_time - start_time

            # 통계 업데이트
            stats['call_count'] += 1
            stats['total_time'] += execution_time
            stats['min_time'] = min(stats['min_time'], execution_time)
            stats['max_time'] = max(stats['max_time'], execution_time)
            stats['average_time'] = stats['total_time'] / stats['call_count']

            print(f"📊 '{func_name}' executed in {execution_time:.4f}s (call #{stats['call_count']})")

            return result

        except Exception as e:
            # 에러가 발생해도 호출 횟수는 증가
            end_time = time.perf_counter()
            execution_time = end_time - start_time

            stats['call_count'] += 1
            stats['total_time'] += execution_time
            stats['min_time'] = min(stats['min_time'], execution_time)
            stats['max_time'] = max(stats['max_time'], execution_time)
            stats['average_time'] = stats['total_time'] / stats['call_count']

            print(f"📊 '{func_name}' failed in {execution_time:.4f}s (call #{stats['call_count']})")
            raise

    return wrapper

def get_performance_stats() -> Dict[str, Dict[str, Any]]:
    """
    수집된 성능 통계를 반환합니다.

    Returns:
        Dict[str, Dict[str, Any]]: 함수별 성능 통계
    """
    # 깊은 복사를 통해 원본 보호
    import copy
    return copy.deepcopy(performance_stats)

def clear_performance_stats() -> None:
    """
    성능 통계를 초기화합니다.
    """
    global performance_stats
    cleared_functions = list(performance_stats.keys())
    performance_stats.clear()
    print(f"📊 Performance statistics cleared for {len(cleared_functions)} functions")

def print_performance_report() -> None:
    """
    성능 통계 보고서를 출력합니다.
    """
    if not performance_stats:
        print("📊 No performance statistics available")
        return

    print("📊 Performance Statistics Report")
    print("=" * 50)

    for func_name, stats in performance_stats.items():
        print(f"\n🔍 Function: {func_name}")
        print(f"   Calls: {stats['call_count']}")
        print(f"   Total Time: {stats['total_time']:.4f}s")
        print(f"   Average Time: {stats['average_time']:.4f}s")
        print(f"   Min Time: {stats['min_time']:.4f}s")
        print(f"   Max Time: {stats['max_time']:.4f}s")
        print(f"   First Called: {stats['first_called']}")
        print(f"   Last Called: {stats['last_called']}")

# 테스트용 Agent 클래스
class TestAgent:
    """
    데코레이터 테스트를 위한 예시 Agent 클래스
    """

    def __init__(self, name: str):
        self.name = name
        print(f"TestAgent '{name}' initialized")

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

    @performance_tracker
    @method_timer
    def combined_decorators_task(self, value: int):
        """여러 데코레이터가 적용된 작업"""
        time.sleep(0.3)
        return value * 2

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
    processing_time = data_size * 0.1
    time.sleep(processing_time)
    return f"Processed {data_size} items in {processing_time}s"

@advanced_timing_decorator(threshold=0.1, show_args=False, show_result=False)
def error_prone_function(should_fail: bool = False):
    """에러가 발생할 수 있는 함수"""
    time.sleep(0.2)
    if should_fail:
        raise ValueError("Intentional error for testing")
    return "Success!"

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

    # 임계값 초과 테스트 (0.5초 > 0.2초 임계값)
    print("--- 임계값 초과 테스트 ---")
    try:
        result = error_prone_function(should_fail=False)
        print(f"결과: {result}")
    except Exception as e:
        print(f"에러 처리됨: {e}")

    # 에러 처리 테스트
    print("\n--- 에러 처리 테스트 ---")
    try:
        error_prone_function(should_fail=True)
    except ValueError:
        print("에러가 올바르게 처리되었습니다.")

    print("\n" + "="*60 + "\n")

    # 클래스 메서드 테스트
    print("--- 클래스 메서드 타이밍 테스트 ---")
    agent = TestAgent("TestBot")

    agent.quick_task()
    print()

    agent.medium_task("데이터 분석")
    print()

    agent.slow_task(0.8)
    print()

    # 여러 데코레이터 조합 테스트
    print("--- 여러 데코레이터 조합 테스트 ---")
    result = agent.combined_decorators_task(42)
    print(f"결과: {result}\n")

    print("="*60 + "\n")

    # 성능 추적 데코레이터 테스트
    print("--- 성능 추적 데코레이터 테스트 ---")

    # 여러 번 실행하여 통계 수집
    print("반복 작업 실행:")
    for i in range(3):
        agent.repeated_task(f"data_{i}")
    print()

    print("데이터 처리 작업 실행:")
    for size in [1, 2, 3]:
        data_processing_function(size)
    print()

    # 동일 함수를 다시 실행
    print("추가 실행:")
    data_processing_function(1)
    agent.repeated_task("final_data")
    print()

    print("="*60 + "\n")

    # 성능 통계 확인
    print("--- 성능 통계 확인 ---")
    stats = get_performance_stats()
    for func_name, stat in stats.items():
        print(f"📋 함수: {func_name}")
        print(f"   호출 횟수: {stat['call_count']}")
        print(f"   총 실행 시간: {stat['total_time']:.4f}초")
        print(f"   평균 실행 시간: {stat['average_time']:.4f}초")
        print(f"   최소 실행 시간: {stat['min_time']:.4f}초")
        print(f"   최대 실행 시간: {stat['max_time']:.4f}초")
        print()

    # 성능 보고서 출력
    print("--- 성능 보고서 ---")
    print_performance_report()

    print("\n" + "="*60 + "\n")

    # 통계 초기화 테스트
    print("--- 통계 초기화 테스트 ---")
    clear_performance_stats()

    # 초기화 후 함수 실행
    print("초기화 후 테스트 실행:")
    data_processing_function(1)
    agent.repeated_task("test_after_clear")

    # 초기화된 통계 확인
    stats = get_performance_stats()
    print(f"\n초기화 후 통계: {len(stats)}개 함수")
    for func_name, stat in stats.items():
        print(f"  📋 {func_name}: {stat['call_count']}회 호출, 평균 {stat['average_time']:.4f}초")

"""
학습 포인트:
1. 데코레이터의 기본 구조와 @functools.wraps 사용법
2. 클로저를 활용한 설정 가능한 데코레이터 구현
3. *args, **kwargs를 사용한 다양한 함수 시그니처 지원
4. 예외 처리를 포함한 견고한 데코레이터 작성
5. 전역 상태를 사용한 통계 수집 패턴
6. 여러 데코레이터의 조합 사용법
7. time.perf_counter()를 사용한 정확한 시간 측정
8. 메타데이터 수집과 로깅 패턴

이러한 데코레이터들은 실제 AI Agent 개발에서 성능 모니터링,
디버깅, 프로파일링에 매우 유용합니다. 특히 API 호출이나
데이터 처리 작업의 성능을 추적할 때 필수적입니다.
"""