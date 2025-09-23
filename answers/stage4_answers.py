"""
4단계: 대용량 데이터 처리 답안 스크립트
"""

import pandas as pd
import numpy as np
import dask.dataframe as dd
import dask
from dask.distributed import Client, as_completed
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil
import time
import gc
from collections import deque
import logging
import threading
import queue

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 문제 1: Dask를 활용한 대용량 데이터 처리
def problem1_solution():
    """Dask를 활용한 대용량 데이터 처리"""

    def create_large_dataset():
        """대용량 데이터셋 생성"""
        logger.info("대용량 샘플 데이터 생성 중...")

        # 실제로는 100GB+ 데이터를 시뮬레이션하기 위해 여러 파일로 분할
        chunk_size = 1000000  # 100만 행씩
        num_chunks = 10  # 총 1000만 행

        for i in range(num_chunks):
            chunk_data = pd.DataFrame({
                'id': range(i * chunk_size, (i + 1) * chunk_size),
                'value1': np.random.randn(chunk_size),
                'value2': np.random.randint(1, 1000, chunk_size),
                'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], chunk_size),
                'timestamp': pd.date_range('2020-01-01', periods=chunk_size, freq='S'),
                'amount': np.random.exponential(100, chunk_size)
            })
            chunk_data.to_csv(f'large_data_chunk_{i}.csv', index=False)

        logger.info(f"{num_chunks}개 청크 파일 생성 완료")
        return num_chunks

    def process_with_dask():
        """Dask를 사용한 대용량 데이터 처리"""
        # Dask 클라이언트 시작
        client = Client(n_workers=4, threads_per_worker=2, memory_limit='2GB')
        logger.info(f"Dask 클라이언트 시작: {client}")

        try:
            # 여러 CSV 파일을 하나의 Dask DataFrame으로 로드
            df = dd.read_csv('large_data_chunk_*.csv')
            logger.info(f"Dask DataFrame 로드 완료: {df.npartitions} 파티션")

            # 메모리 사용량 모니터링
            initial_memory = psutil.virtual_memory().used / 1024**3
            logger.info(f"초기 메모리 사용량: {initial_memory:.2f} GB")

            # 병렬 그룹화 및 집계
            logger.info("그룹화 및 집계 시작...")

            # 카테고리별 집계
            category_stats = df.groupby('category').agg({
                'value1': ['mean', 'std'],
                'value2': ['sum', 'count'],
                'amount': ['sum', 'mean', 'max']
            }).compute()

            logger.info("카테고리별 집계 완료")
            print(category_stats)

            # 시간별 집계
            df['hour'] = dd.to_datetime(df['timestamp']).dt.hour
            hourly_stats = df.groupby('hour')['amount'].agg(['sum', 'mean', 'count']).compute()

            logger.info("시간별 집계 완료")

            # 복잡한 계산 수행
            df['value_ratio'] = df['value1'] / df['value2']
            df['amount_log'] = dd.log(df['amount'] + 1)

            # 결과를 파티션별로 저장
            result_df = df[['id', 'category', 'value_ratio', 'amount_log', 'hour']]
            result_df.to_csv('processed_large_data_*.csv', index=False)

            logger.info("결과 저장 완료")

            # 최종 메모리 사용량
            final_memory = psutil.virtual_memory().used / 1024**3
            logger.info(f"최종 메모리 사용량: {final_memory:.2f} GB")
            logger.info(f"메모리 증가량: {final_memory - initial_memory:.2f} GB")

            return category_stats, hourly_stats

        finally:
            client.close()

    # 대용량 데이터 처리 실행
    num_chunks = create_large_dataset()
    category_stats, hourly_stats = process_with_dask()

    return category_stats, hourly_stats

# 문제 2: 멀티프로세싱을 활용한 데이터 처리
def problem2_solution():
    """멀티프로세싱을 활용한 데이터 처리"""

    def process_chunk(chunk_info):
        """개별 청크 처리 함수"""
        chunk_id, chunk_data = chunk_info
        process_id = mp.current_process().pid

        # 무거운 계산 시뮬레이션
        result = {
            'chunk_id': chunk_id,
            'process_id': process_id,
            'mean_value': chunk_data['value'].mean(),
            'std_value': chunk_data['value'].std(),
            'sum_value': chunk_data['value'].sum(),
            'processed_rows': len(chunk_data)
        }

        # 인위적 지연 (복잡한 처리 시뮬레이션)
        time.sleep(0.5)

        return result

    def multiprocess_data_processing():
        """멀티프로세싱으로 데이터 처리"""
        # 대용량 데이터 생성
        total_size = 1000000
        chunk_size = 100000
        num_chunks = total_size // chunk_size

        logger.info(f"총 {num_chunks}개 청크 생성 중...")

        chunks = []
        for i in range(num_chunks):
            chunk_data = pd.DataFrame({
                'id': range(i * chunk_size, (i + 1) * chunk_size),
                'value': np.random.randn(chunk_size),
                'category': np.random.choice(['A', 'B', 'C'], chunk_size)
            })
            chunks.append((i, chunk_data))

        # 프로세스 풀을 사용한 병렬 처리
        num_workers = min(mp.cpu_count(), 4)
        logger.info(f"{num_workers}개 프로세스로 병렬 처리 시작")

        # 진행률 추적을 위한 큐
        progress_queue = mp.Queue()

        def progress_tracker(total_tasks):
            """진행률 추적 스레드"""
            completed = 0
            while completed < total_tasks:
                try:
                    progress_queue.get(timeout=1)
                    completed += 1
                    logger.info(f"진행률: {completed}/{total_tasks} ({completed/total_tasks*100:.1f}%)")
                except:
                    pass

        # 진행률 추적 스레드 시작
        progress_thread = threading.Thread(target=progress_tracker, args=(num_chunks,))
        progress_thread.start()

        results = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # 모든 청크를 처리할 future 생성
            future_to_chunk = {executor.submit(process_chunk, chunk): chunk for chunk in chunks}

            # 완료된 작업 처리
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    result = future.result()
                    results.append(result)
                    progress_queue.put(1)  # 진행률 업데이트
                except Exception as exc:
                    logger.error(f'청크 {chunk[0]} 처리 중 오류 발생: {exc}')

        # 진행률 추적 스레드 종료
        progress_thread.join()

        # 결과 병합
        results_df = pd.DataFrame(results)
        logger.info("모든 청크 처리 완료")

        # 메모리 효율적인 최종 집계
        final_stats = {
            'total_processed_rows': results_df['processed_rows'].sum(),
            'overall_mean': (results_df['mean_value'] * results_df['processed_rows']).sum() / results_df['processed_rows'].sum(),
            'processes_used': results_df['process_id'].nunique()
        }

        logger.info(f"최종 통계: {final_stats}")
        return results_df, final_stats

    return multiprocess_data_processing()

# 문제 3: 스트리밍 데이터 처리
def problem3_solution():
    """스트리밍 데이터 처리"""

    class StreamProcessor:
        def __init__(self, window_size=100, threshold=3.0):
            self.window_size = window_size
            self.threshold = threshold
            self.data_window = deque(maxlen=window_size)
            self.alerts = []

        def process_streaming_data(self, duration_seconds=30):
            """스트리밍 데이터 처리 시뮬레이션"""
            logger.info(f"{duration_seconds}초 동안 스트리밍 데이터 처리 시작")

            start_time = time.time()
            data_points_processed = 0

            while time.time() - start_time < duration_seconds:
                # 새로운 데이터 포인트 생성 (실시간 데이터 시뮬레이션)
                # 가끔 이상치 포함
                if np.random.random() < 0.05:  # 5% 확률로 이상치
                    value = np.random.normal(50, 5)  # 이상치
                else:
                    value = np.random.normal(10, 2)  # 정상 데이터

                timestamp = time.time()
                data_point = {
                    'timestamp': timestamp,
                    'value': value,
                    'id': data_points_processed
                }

                # 슬라이딩 윈도우에 추가
                self.data_window.append(data_point)
                data_points_processed += 1

                # 윈도우가 충분히 찬 경우 분석 수행
                if len(self.data_window) >= 10:
                    self._analyze_window()

                # 처리 간격 (100ms)
                time.sleep(0.1)

            logger.info(f"총 {data_points_processed}개 데이터 포인트 처리 완료")
            logger.info(f"총 {len(self.alerts)}개 알림 생성")

            return data_points_processed, self.alerts

        def _analyze_window(self):
            """현재 윈도우 분석"""
            if len(self.data_window) < 10:
                return

            # 최근 10개 데이터 포인트로 통계 계산
            recent_values = [point['value'] for point in list(self.data_window)[-10:]]
            mean_val = np.mean(recent_values)
            std_val = np.std(recent_values)

            # 최신 값이 이상치인지 확인
            latest_point = self.data_window[-1]
            if abs(latest_point['value'] - mean_val) > self.threshold * std_val:
                alert = {
                    'timestamp': latest_point['timestamp'],
                    'value': latest_point['value'],
                    'mean': mean_val,
                    'std': std_val,
                    'z_score': (latest_point['value'] - mean_val) / std_val if std_val > 0 else 0
                }
                self.alerts.append(alert)
                logger.warning(f"이상치 탐지: 값={latest_point['value']:.2f}, Z-score={alert['z_score']:.2f}")

        def _sliding_window_aggregation(self):
            """슬라이딩 윈도우 집계"""
            if len(self.data_window) < self.window_size:
                return None

            values = [point['value'] for point in self.data_window]
            return {
                'window_mean': np.mean(values),
                'window_std': np.std(values),
                'window_min': np.min(values),
                'window_max': np.max(values),
                'data_points': len(values)
            }

    # 스트리밍 처리 실행
    processor = StreamProcessor(window_size=50, threshold=2.5)
    processed_count, alerts = processor.process_streaming_data(duration_seconds=10)

    return processed_count, alerts

# 문제 4: 메모리 최적화
def problem4_solution():
    """메모리 최적화"""

    def create_memory_intensive_data():
        """메모리 집약적 데이터 생성"""
        logger.info("메모리 집약적 데이터 생성")

        # 비효율적인 데이터 타입으로 시작
        df = pd.DataFrame({
            'id': range(1000000),  # int64 (8 bytes)
            'category': [f'Category_{i%100}' for i in range(1000000)],  # object (string)
            'value': np.random.randn(1000000),  # float64 (8 bytes)
            'flag': np.random.choice([True, False], 1000000),  # bool이지만 object로 저장될 수 있음
            'small_int': np.random.randint(0, 100, 1000000),  # int64이지만 작은 값
        })

        return df

    def profile_memory_usage(df, stage_name):
        """메모리 사용량 프로파일링"""
        memory_usage = df.memory_usage(deep=True)
        total_memory = memory_usage.sum() / 1024**2  # MB

        logger.info(f"{stage_name} 메모리 사용량:")
        for col, usage in memory_usage.items():
            if col != 'Index':
                logger.info(f"  {col}: {usage / 1024**2:.2f} MB ({df[col].dtype})")
        logger.info(f"  총 메모리 사용량: {total_memory:.2f} MB")

        return total_memory

    def optimize_data_types(df):
        """데이터 타입 최적화"""
        df_optimized = df.copy()

        # 정수 컬럼 최적화
        for col in ['id', 'small_int']:
            if col in df_optimized.columns:
                max_val = df_optimized[col].max()
                min_val = df_optimized[col].min()

                if min_val >= 0:
                    if max_val < 255:
                        df_optimized[col] = df_optimized[col].astype('uint8')
                    elif max_val < 65535:
                        df_optimized[col] = df_optimized[col].astype('uint16')
                    elif max_val < 4294967295:
                        df_optimized[col] = df_optimized[col].astype('uint32')
                else:
                    if min_val >= -128 and max_val <= 127:
                        df_optimized[col] = df_optimized[col].astype('int8')
                    elif min_val >= -32768 and max_val <= 32767:
                        df_optimized[col] = df_optimized[col].astype('int16')
                    elif min_val >= -2147483648 and max_val <= 2147483647:
                        df_optimized[col] = df_optimized[col].astype('int32')

        # 카테고리 데이터 최적화
        if 'category' in df_optimized.columns:
            df_optimized['category'] = df_optimized['category'].astype('category')

        # 부울 데이터 최적화
        if 'flag' in df_optimized.columns:
            df_optimized['flag'] = df_optimized['flag'].astype('bool')

        # 실수 데이터 최적화 (정밀도 허용 범위 내에서)
        if 'value' in df_optimized.columns:
            df_optimized['value'] = df_optimized['value'].astype('float32')

        return df_optimized

    def chunked_processing_demo():
        """청크 단위 처리 데모"""
        logger.info("청크 단위 처리 시작")

        chunk_size = 100000
        total_sum = 0
        total_count = 0

        # 메모리 제한 시뮬레이션 (실제로는 더 큰 파일)
        for chunk_id in range(10):
            # 청크 로드
            chunk_start = chunk_id * chunk_size
            chunk_end = (chunk_id + 1) * chunk_size

            chunk_data = pd.DataFrame({
                'id': range(chunk_start, chunk_end),
                'value': np.random.randn(chunk_size)
            })

            # 청크 처리
            chunk_sum = chunk_data['value'].sum()
            chunk_count = len(chunk_data)

            total_sum += chunk_sum
            total_count += chunk_count

            # 메모리 정리
            del chunk_data
            gc.collect()

            if chunk_id % 3 == 0:
                current_memory = psutil.virtual_memory().percent
                logger.info(f"청크 {chunk_id} 처리 완료, 메모리 사용률: {current_memory:.1f}%")

        overall_mean = total_sum / total_count
        logger.info(f"전체 평균: {overall_mean:.6f}")

        return overall_mean

    # 메모리 최적화 실행
    df_original = create_memory_intensive_data()
    original_memory = profile_memory_usage(df_original, "최적화 전")

    df_optimized = optimize_data_types(df_original)
    optimized_memory = profile_memory_usage(df_optimized, "최적화 후")

    memory_reduction = ((original_memory - optimized_memory) / original_memory) * 100
    logger.info(f"메모리 사용량 감소: {memory_reduction:.1f}%")

    # 청크 단위 처리 데모
    chunked_result = chunked_processing_demo()

    return {
        'original_memory': original_memory,
        'optimized_memory': optimized_memory,
        'memory_reduction_percent': memory_reduction,
        'chunked_result': chunked_result
    }

# 문제 5: 분산 처리 시뮬레이션
def problem5_solution():
    """분산 처리 시뮬레이션"""

    class DistributedProcessor:
        def __init__(self, num_nodes=4):
            self.num_nodes = num_nodes
            self.node_status = {f'node_{i}': 'active' for i in range(num_nodes)}
            self.task_results = {}

        def map_reduce_simulation(self, data_size=1000000):
            """Map-Reduce 패턴 시뮬레이션"""
            logger.info(f"Map-Reduce 시뮬레이션 시작 (데이터 크기: {data_size})")

            # 1. 데이터 분할 (Map 단계)
            chunk_size = data_size // self.num_nodes
            data_chunks = []

            for i in range(self.num_nodes):
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size if i < self.num_nodes - 1 else data_size

                chunk = {
                    'node_id': f'node_{i}',
                    'data': np.random.randn(end_idx - start_idx),
                    'start_idx': start_idx,
                    'end_idx': end_idx
                }
                data_chunks.append(chunk)

            logger.info(f"데이터를 {len(data_chunks)}개 청크로 분할")

            # 2. Map 작업 실행
            map_results = []
            with ProcessPoolExecutor(max_workers=self.num_nodes) as executor:
                future_to_chunk = {
                    executor.submit(self._map_task, chunk): chunk
                    for chunk in data_chunks
                }

                for future in as_completed(future_to_chunk):
                    chunk = future_to_chunk[future]
                    try:
                        result = future.result()
                        map_results.append(result)
                        logger.info(f"{chunk['node_id']} Map 작업 완료")
                    except Exception as exc:
                        logger.error(f"{chunk['node_id']} Map 작업 실패: {exc}")
                        # 장애 처리: 다른 노드에서 재실행
                        self._handle_node_failure(chunk['node_id'])

            # 3. Reduce 작업 실행
            final_result = self._reduce_task(map_results)
            logger.info(f"Reduce 작업 완료: {final_result}")

            return final_result

        def _map_task(self, chunk):
            """Map 작업 (각 노드에서 실행)"""
            node_id = chunk['node_id']

            # 장애 시뮬레이션 (5% 확률)
            if np.random.random() < 0.05:
                raise Exception(f"{node_id} 하드웨어 장애 발생")

            # 실제 Map 작업: 통계 계산
            data = chunk['data']
            result = {
                'node_id': node_id,
                'count': len(data),
                'sum': np.sum(data),
                'sum_squares': np.sum(data**2),
                'min': np.min(data),
                'max': np.max(data)
            }

            # 처리 시간 시뮬레이션
            time.sleep(np.random.uniform(0.1, 0.5))

            return result

        def _reduce_task(self, map_results):
            """Reduce 작업 (결과 집계)"""
            if not map_results:
                return None

            total_count = sum(r['count'] for r in map_results)
            total_sum = sum(r['sum'] for r in map_results)
            total_sum_squares = sum(r['sum_squares'] for r in map_results)
            global_min = min(r['min'] for r in map_results)
            global_max = max(r['max'] for r in map_results)

            global_mean = total_sum / total_count
            global_variance = (total_sum_squares / total_count) - (global_mean ** 2)
            global_std = np.sqrt(global_variance)

            return {
                'total_count': total_count,
                'global_mean': global_mean,
                'global_std': global_std,
                'global_min': global_min,
                'global_max': global_max,
                'nodes_used': len(map_results)
            }

        def _handle_node_failure(self, failed_node):
            """노드 장애 처리"""
            logger.warning(f"{failed_node} 장애 감지")
            self.node_status[failed_node] = 'failed'

            # 복구 로직 (간단한 예시)
            time.sleep(1)  # 복구 시간
            self.node_status[failed_node] = 'recovered'
            logger.info(f"{failed_node} 복구 완료")

        def performance_monitoring(self):
            """성능 모니터링"""
            metrics = {
                'active_nodes': sum(1 for status in self.node_status.values() if status == 'active'),
                'failed_nodes': sum(1 for status in self.node_status.values() if status == 'failed'),
                'memory_usage': psutil.virtual_memory().percent,
                'cpu_usage': psutil.cpu_percent(),
                'timestamp': time.time()
            }

            logger.info(f"성능 메트릭: {metrics}")
            return metrics

    # 분산 처리 시뮬레이션 실행
    processor = DistributedProcessor(num_nodes=4)

    # 성능 모니터링 시작
    initial_metrics = processor.performance_monitoring()

    # Map-Reduce 실행
    result = processor.map_reduce_simulation(data_size=500000)

    # 최종 성능 메트릭
    final_metrics = processor.performance_monitoring()

    return result, initial_metrics, final_metrics

if __name__ == "__main__":
    print("=== 문제 1: Dask를 활용한 대용량 데이터 처리 ===")
    try:
        category_stats, hourly_stats = problem1_solution()
        print("Dask 처리 완료")
    except Exception as e:
        print(f"Dask 처리 중 오류: {e}")

    print("\n=== 문제 2: 멀티프로세싱을 활용한 데이터 처리 ===")
    results_df, final_stats = problem2_solution()
    print(f"멀티프로세싱 결과: {final_stats}")

    print("\n=== 문제 3: 스트리밍 데이터 처리 ===")
    processed_count, alerts = problem3_solution()
    print(f"처리된 데이터 포인트: {processed_count}, 알림 개수: {len(alerts)}")

    print("\n=== 문제 4: 메모리 최적화 ===")
    optimization_results = problem4_solution()
    print(f"메모리 최적화 결과: {optimization_results}")

    print("\n=== 문제 5: 분산 처리 시뮬레이션 ===")
    distributed_result, initial_metrics, final_metrics = problem5_solution()
    print(f"분산 처리 결과: {distributed_result}")