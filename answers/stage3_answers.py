"""
3단계: 데이터베이스 연동 답안 스크립트
"""

import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine, text
import logging
import os
from datetime import datetime
import time

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 문제 1: SQLite 데이터베이스 연동
def problem1_solution():
    """SQLite 데이터베이스 연동"""
    # SQLite 데이터베이스 연결
    db_path = 'sample_database.db'
    engine = create_engine(f'sqlite:///{db_path}')

    # 샘플 DataFrame 생성
    df = pd.DataFrame({
        'customer_id': range(1, 101),
        'name': [f'Customer_{i}' for i in range(1, 101)],
        'age': np.random.randint(18, 80, 100),
        'city': np.random.choice(['Seoul', 'Busan', 'Incheon', 'Daegu'], 100),
        'purchase_amount': np.random.randint(1000, 100000, 100)
    })

    # DataFrame을 테이블로 저장
    df.to_sql('customers', engine, if_exists='replace', index=False)
    logger.info(f"테이블 'customers'에 {len(df)}개 레코드 저장 완료")

    # SQL 쿼리로 데이터 조회
    with engine.connect() as conn:
        # 전체 데이터 조회
        result = pd.read_sql("SELECT * FROM customers LIMIT 5", conn)
        print("전체 데이터 (상위 5개):")
        print(result)

        # 조건부 데이터 추출
        high_value_customers = pd.read_sql("""
            SELECT name, age, city, purchase_amount
            FROM customers
            WHERE purchase_amount > 50000
            ORDER BY purchase_amount DESC
        """, conn)

        print(f"\n고액 구매 고객 ({len(high_value_customers)}명):")
        print(high_value_customers.head())

        # 도시별 통계
        city_stats = pd.read_sql("""
            SELECT city,
                   COUNT(*) as customer_count,
                   AVG(age) as avg_age,
                   AVG(purchase_amount) as avg_purchase
            FROM customers
            GROUP BY city
        """, conn)

        print("\n도시별 통계:")
        print(city_stats)

    return df

# 문제 2: ETL 파이프라인 구현
def problem2_solution():
    """ETL 파이프라인 구현"""

    def extract_data():
        """데이터 추출"""
        logger.info("데이터 추출 시작")

        # 여러 소스에서 데이터 추출 시뮬레이션
        source1 = pd.DataFrame({
            'product_id': range(1, 51),
            'product_name': [f'Product_{i}' for i in range(1, 51)],
            'category': np.random.choice(['A', 'B', 'C'], 50),
            'price': np.random.randint(1000, 50000, 50)
        })

        source2 = pd.DataFrame({
            'order_id': range(1, 201),
            'product_id': np.random.randint(1, 51, 200),
            'quantity': np.random.randint(1, 10, 200),
            'order_date': pd.date_range('2023-01-01', periods=200, freq='D')
        })

        logger.info(f"추출 완료: 제품 {len(source1)}개, 주문 {len(source2)}개")
        return source1, source2

    def transform_data(products, orders):
        """데이터 변환"""
        logger.info("데이터 변환 시작")

        # 제품 데이터 변환
        products['price_category'] = pd.cut(products['price'],
                                          bins=[0, 10000, 30000, float('inf')],
                                          labels=['Low', 'Medium', 'High'])

        # 주문 데이터 변환
        orders['total_amount'] = orders['quantity']  # 실제로는 가격과 곱해야 함
        orders['month'] = orders['order_date'].dt.month
        orders['weekday'] = orders['order_date'].dt.dayofweek

        # 조인 수행
        merged_data = orders.merge(products, on='product_id', how='left')

        # 집계 데이터 생성
        daily_summary = merged_data.groupby(orders['order_date'].dt.date).agg({
            'order_id': 'count',
            'quantity': 'sum',
            'price': 'mean'
        }).reset_index()
        daily_summary.columns = ['date', 'order_count', 'total_quantity', 'avg_price']

        logger.info("데이터 변환 완료")
        return merged_data, daily_summary

    def load_data(merged_data, daily_summary):
        """데이터 적재"""
        logger.info("데이터 적재 시작")

        engine = create_engine('sqlite:///etl_database.db')

        # 변환된 데이터 저장
        merged_data.to_sql('order_details', engine, if_exists='replace', index=False)
        daily_summary.to_sql('daily_summary', engine, if_exists='replace', index=False)

        logger.info(f"적재 완료: 주문상세 {len(merged_data)}개, 일별요약 {len(daily_summary)}개")

    # ETL 파이프라인 실행
    try:
        products, orders = extract_data()
        merged_data, daily_summary = transform_data(products, orders)
        load_data(merged_data, daily_summary)
        logger.info("ETL 파이프라인 성공적으로 완료")
    except Exception as e:
        logger.error(f"ETL 파이프라인 실행 중 오류 발생: {e}")
        raise

    return merged_data, daily_summary

# 문제 3: 배치 프로세싱
def problem3_solution():
    """배치 프로세싱"""

    def create_large_csv():
        """대용량 CSV 파일 생성"""
        logger.info("대용량 샘플 데이터 생성")
        large_df = pd.DataFrame({
            'id': range(1, 100001),
            'value1': np.random.randn(100000),
            'value2': np.random.randint(1, 1000, 100000),
            'category': np.random.choice(['A', 'B', 'C', 'D'], 100000),
            'date': pd.date_range('2020-01-01', periods=100000, freq='H')
        })
        large_df.to_csv('large_data.csv', index=False)
        return 'large_data.csv'

    def process_in_chunks(file_path, chunk_size=10000):
        """청크 단위로 처리"""
        engine = create_engine('sqlite:///batch_database.db')

        total_rows = sum(1 for line in open(file_path)) - 1  # 헤더 제외
        processed_rows = 0

        # 체크포인트 파일
        checkpoint_file = 'processing_checkpoint.txt'
        start_chunk = 0

        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                start_chunk = int(f.read().strip())
            logger.info(f"체크포인트에서 재시작: 청크 {start_chunk}")

        chunk_iterator = pd.read_csv(file_path, chunksize=chunk_size)

        for i, chunk in enumerate(chunk_iterator):
            if i < start_chunk:
                continue

            try:
                # 데이터 변환
                chunk['value1_squared'] = chunk['value1'] ** 2
                chunk['value_ratio'] = chunk['value1'] / chunk['value2']
                chunk['month'] = pd.to_datetime(chunk['date']).dt.month

                # 데이터베이스에 저장
                chunk.to_sql('processed_data', engine, if_exists='append', index=False)

                processed_rows += len(chunk)
                progress = (processed_rows / total_rows) * 100

                logger.info(f"청크 {i+1} 처리 완료 ({processed_rows}/{total_rows} - {progress:.1f}%)")

                # 체크포인트 업데이트
                with open(checkpoint_file, 'w') as f:
                    f.write(str(i+1))

                # 인위적 딜레이 (실제 처리 시간 시뮬레이션)
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"청크 {i+1} 처리 중 오류: {e}")
                raise

        # 처리 완료 후 체크포인트 파일 삭제
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)

        logger.info("배치 처리 완료")
        return processed_rows

    # 대용량 파일 생성 및 처리
    file_path = create_large_csv()
    processed_count = process_in_chunks(file_path)

    return processed_count

# 문제 4: 데이터 동기화
def problem4_solution():
    """데이터 동기화"""

    def create_source_target_dbs():
        """소스 및 타겟 데이터베이스 생성"""
        # 소스 데이터베이스
        source_engine = create_engine('sqlite:///source_db.db')
        source_data = pd.DataFrame({
            'id': range(1, 101),
            'name': [f'Item_{i}' for i in range(1, 101)],
            'value': np.random.randint(1, 1000, 100),
            'last_modified': pd.date_range('2023-01-01', periods=100, freq='H')
        })
        source_data.to_sql('items', source_engine, if_exists='replace', index=False)

        # 타겟 데이터베이스 (일부 데이터만)
        target_engine = create_engine('sqlite:///target_db.db')
        target_data = source_data.iloc[:80].copy()  # 처음 80개만
        target_data.loc[10:19, 'value'] = target_data.loc[10:19, 'value'] + 100  # 일부 값 변경
        target_data.to_sql('items', target_engine, if_exists='replace', index=False)

        return source_engine, target_engine

    def sync_databases(source_engine, target_engine):
        """데이터베이스 동기화"""
        sync_log = []

        with source_engine.connect() as source_conn, target_engine.connect() as target_conn:
            # 소스 데이터 조회
            source_df = pd.read_sql("SELECT * FROM items", source_conn)
            logger.info(f"소스 데이터: {len(source_df)}개 레코드")

            # 타겟 데이터 조회
            try:
                target_df = pd.read_sql("SELECT * FROM items", target_conn)
                logger.info(f"타겟 데이터: {len(target_df)}개 레코드")
            except:
                target_df = pd.DataFrame()
                logger.info("타겟 테이블이 존재하지 않음")

            if target_df.empty:
                # 전체 데이터 복사
                source_df.to_sql('items', target_engine, if_exists='replace', index=False)
                sync_log.append({'action': 'full_copy', 'count': len(source_df)})
            else:
                # 증분 업데이트
                source_df['last_modified'] = pd.to_datetime(source_df['last_modified'])
                target_df['last_modified'] = pd.to_datetime(target_df['last_modified'])

                # 새로운 레코드 찾기
                new_records = source_df[~source_df['id'].isin(target_df['id'])]
                if len(new_records) > 0:
                    new_records.to_sql('items', target_engine, if_exists='append', index=False)
                    sync_log.append({'action': 'insert', 'count': len(new_records)})

                # 변경된 레코드 찾기
                merged = source_df.merge(target_df, on='id', suffixes=('_source', '_target'))
                changed_records = merged[
                    (merged['value_source'] != merged['value_target']) |
                    (merged['last_modified_source'] > merged['last_modified_target'])
                ]

                if len(changed_records) > 0:
                    for _, record in changed_records.iterrows():
                        target_conn.execute(text("""
                            UPDATE items
                            SET value = :value, last_modified = :last_modified
                            WHERE id = :id
                        """), {
                            'value': record['value_source'],
                            'last_modified': record['last_modified_source'],
                            'id': record['id']
                        })
                    target_conn.commit()
                    sync_log.append({'action': 'update', 'count': len(changed_records)})

        return sync_log

    # 동기화 실행
    source_engine, target_engine = create_source_target_dbs()
    sync_result = sync_databases(source_engine, target_engine)

    logger.info("동기화 결과:")
    for log_entry in sync_result:
        logger.info(f"  {log_entry['action']}: {log_entry['count']}개 레코드")

    return sync_result

# 문제 5: 복잡한 데이터 변환
def problem5_solution():
    """복잡한 데이터 변환"""

    def create_complex_data():
        """복잡한 샘플 데이터 생성"""
        engine = create_engine('sqlite:///complex_db.db')

        # 여러 테이블 생성
        customers = pd.DataFrame({
            'customer_id': range(1, 101),
            'name': [f'Customer_{i}' for i in range(1, 101)],
            'region': np.random.choice(['North', 'South', 'East', 'West'], 100)
        })

        products = pd.DataFrame({
            'product_id': range(1, 51),
            'product_name': [f'Product_{i}' for i in range(1, 51)],
            'category_id': np.random.randint(1, 6, 50)
        })

        categories = pd.DataFrame({
            'category_id': range(1, 6),
            'category_name': ['Electronics', 'Clothing', 'Food', 'Books', 'Sports']
        })

        orders = pd.DataFrame({
            'order_id': range(1, 1001),
            'customer_id': np.random.randint(1, 101, 1000),
            'product_id': np.random.randint(1, 51, 1000),
            'quantity': np.random.randint(1, 10, 1000),
            'order_date': pd.date_range('2023-01-01', periods=1000, freq='H'),
            'amount': np.random.randint(1000, 100000, 1000)
        })

        # 계층적 데이터 (조직도)
        hierarchy = pd.DataFrame({
            'employee_id': range(1, 21),
            'manager_id': [None, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10],
            'name': [f'Employee_{i}' for i in range(1, 21)],
            'department': np.random.choice(['Sales', 'Marketing', 'IT', 'HR'], 20)
        })

        # 데이터베이스에 저장
        customers.to_sql('customers', engine, if_exists='replace', index=False)
        products.to_sql('products', engine, if_exists='replace', index=False)
        categories.to_sql('categories', engine, if_exists='replace', index=False)
        orders.to_sql('orders', engine, if_exists='replace', index=False)
        hierarchy.to_sql('hierarchy', engine, if_exists='replace', index=False)

        return engine

    def complex_transformations(engine):
        """복잡한 데이터 변환 수행"""

        with engine.connect() as conn:
            # 1. 여러 테이블 조인
            complex_join = pd.read_sql("""
                SELECT
                    o.order_id,
                    c.name as customer_name,
                    c.region,
                    p.product_name,
                    cat.category_name,
                    o.quantity,
                    o.amount,
                    o.order_date
                FROM orders o
                JOIN customers c ON o.customer_id = c.customer_id
                JOIN products p ON o.product_id = p.product_id
                JOIN categories cat ON p.category_id = cat.category_id
            """, conn)

            logger.info(f"조인 결과: {len(complex_join)}개 레코드")

            # 2. 계층적 데이터 평면화
            hierarchy_df = pd.read_sql("SELECT * FROM hierarchy", conn)

            def flatten_hierarchy(df, employee_id, level=0):
                """재귀적으로 계층 구조 평면화"""
                result = []
                employee = df[df['employee_id'] == employee_id].iloc[0]
                result.append({
                    'employee_id': employee_id,
                    'name': employee['name'],
                    'department': employee['department'],
                    'level': level,
                    'path': f"Level_{level}_{employee['name']}"
                })

                subordinates = df[df['manager_id'] == employee_id]
                for _, subordinate in subordinates.iterrows():
                    result.extend(flatten_hierarchy(df, subordinate['employee_id'], level + 1))

                return result

            # 최고 관리자 찾기 (manager_id가 null인 직원)
            top_managers = hierarchy_df[hierarchy_df['manager_id'].isnull()]
            flattened_hierarchy = []

            for _, manager in top_managers.iterrows():
                flattened_hierarchy.extend(flatten_hierarchy(hierarchy_df, manager['employee_id']))

            hierarchy_flat = pd.DataFrame(flattened_hierarchy)
            logger.info(f"평면화된 계층 구조: {len(hierarchy_flat)}개 노드")

            # 3. 피벗 테이블 생성
            pivot_data = complex_join.groupby(['region', 'category_name'])['amount'].sum().reset_index()
            pivot_table = pivot_data.pivot(index='region', columns='category_name', values='amount').fillna(0)
            logger.info(f"피벗 테이블 크기: {pivot_table.shape}")

            # 4. 시계열 데이터 집계
            complex_join['order_date'] = pd.to_datetime(complex_join['order_date'])
            complex_join['month'] = complex_join['order_date'].dt.to_period('M')

            monthly_metrics = complex_join.groupby('month').agg({
                'order_id': 'count',
                'amount': ['sum', 'mean'],
                'quantity': 'sum'
            }).reset_index()

            monthly_metrics.columns = ['month', 'order_count', 'total_amount', 'avg_amount', 'total_quantity']
            logger.info(f"월별 집계: {len(monthly_metrics)}개 월")

        return complex_join, hierarchy_flat, pivot_table, monthly_metrics

    # 복잡한 변환 실행
    engine = create_complex_data()
    join_result, hierarchy_result, pivot_result, timeseries_result = complex_transformations(engine)

    print("피벗 테이블 결과:")
    print(pivot_result)
    print("\n월별 집계 결과:")
    print(timeseries_result.head())

    return join_result, hierarchy_result, pivot_result, timeseries_result

if __name__ == "__main__":
    print("=== 문제 1: SQLite 데이터베이스 연동 ===")
    df1 = problem1_solution()

    print("\n=== 문제 2: ETL 파이프라인 구현 ===")
    merged, summary = problem2_solution()

    print("\n=== 문제 3: 배치 프로세싱 ===")
    processed = problem3_solution()
    print(f"처리된 레코드 수: {processed}")

    print("\n=== 문제 4: 데이터 동기화 ===")
    sync_result = problem4_solution()

    print("\n=== 문제 5: 복잡한 데이터 변환 ===")
    results = problem5_solution()