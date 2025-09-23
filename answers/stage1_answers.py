"""
1단계: 기초 데이터 처리 답안 스크립트
"""

import pandas as pd
import numpy as np
from datetime import datetime

# 문제 1: CSV 파일 읽기 및 기본 정보 확인
def problem1_solution():
    """CSV 파일 읽기 및 기본 정보 확인"""
    # 샘플 데이터 생성 (실제로는 파일에서 읽어옴)
    df = pd.DataFrame({
        'customer_id': range(1, 101),
        'name': [f'Customer_{i}' for i in range(1, 101)],
        'age': np.random.randint(18, 80, 100),
        'city': np.random.choice(['Seoul', 'Busan', 'Incheon'], 100)
    })

    # df = pd.read_csv('customers.csv')  # 실제 파일 읽기

    print("데이터 shape:", df.shape)
    print("\n컬럼명:", df.columns.tolist())
    print("\n처음 5행:")
    print(df.head())
    print("\n기본 통계정보:")
    print(df.describe())

    return df

# 문제 2: 결측값 처리
def problem2_solution():
    """결측값 처리"""
    # 샘플 데이터 생성 (결측값 포함)
    df = pd.DataFrame({
        'product_id': range(1, 201),
        'price': np.random.choice([10.5, 20.0, np.nan, 15.5, np.nan], 200),
        'category': np.random.choice(['A', 'B', np.nan, 'C'], 200),
        'stock': np.random.choice([10, 20, np.nan, np.nan, np.nan], 200)
    })

    print("원본 데이터 결측값 개수:")
    print(df.isnull().sum())

    # price 컬럼 결측값을 평균으로 채우기
    df['price'].fillna(df['price'].mean(), inplace=True)

    # category 컬럼 결측값을 'Unknown'으로 채우기
    df['category'].fillna('Unknown', inplace=True)

    # 결측값이 3개 이상인 행 삭제
    df_cleaned = df.dropna(thresh=len(df.columns) - 2)

    print("\n처리 후 결측값 개수:")
    print(df_cleaned.isnull().sum())
    print(f"\n제거된 행 수: {len(df) - len(df_cleaned)}")

    return df_cleaned

# 문제 3: 데이터 타입 변환
def problem3_solution():
    """데이터 타입 변환"""
    # 샘플 데이터 생성
    df = pd.DataFrame({
        'product_id': range(1, 51),
        'date': ['2023-01-01', '2023-01-02', '2023-01-03'] * 17 + ['2023-01-01'],
        'price': ['10.5', '20.0', '15.5'] * 17 + ['10.5'],
        'is_available': ['True', 'False', 'True'] * 17 + ['True']
    })

    print("변환 전 데이터 타입:")
    print(df.dtypes)

    # 타입 변환
    df['date'] = pd.to_datetime(df['date'])
    df['price'] = df['price'].astype(float)
    df['is_available'] = df['is_available'].astype(bool)

    print("\n변환 후 데이터 타입:")
    print(df.dtypes)

    return df

# 문제 4: 데이터 필터링 및 정렬
def problem4_solution():
    """데이터 필터링 및 정렬"""
    # 샘플 데이터 생성
    dates = pd.date_range('2022-01-01', '2024-12-31', freq='D')
    df = pd.DataFrame({
        'order_id': range(1, len(dates) + 1),
        'date': dates,
        'amount': np.random.randint(100, 5000, len(dates))
    })

    print(f"원본 데이터 크기: {len(df)}")

    # 2023년 데이터만 필터링
    df_2023 = df[df['date'].dt.year == 2023]

    # 주문 금액이 1000 이상인 데이터 필터링
    df_filtered = df_2023[df_2023['amount'] >= 1000]

    # 날짜 순으로 정렬
    df_sorted = df_filtered.sort_values('date')

    print(f"필터링 후 데이터 크기: {len(df_sorted)}")

    # CSV 파일로 저장
    # df_sorted.to_csv('filtered_orders.csv', index=False)

    return df_sorted

# 문제 5: 그룹화 및 집계
def problem5_solution():
    """그룹화 및 집계"""
    # 샘플 데이터 생성
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    df = pd.DataFrame({
        'transaction_id': range(1, 1001),
        'customer_id': np.random.randint(1, 101, 1000),
        'date': np.random.choice(dates, 1000),
        'amount': np.random.randint(10, 1000, 1000),
        'category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], 1000),
        'quantity': np.random.randint(1, 10, 1000)
    })

    # 고객별 총 구매 금액
    customer_total = df.groupby('customer_id')['amount'].sum().reset_index()
    customer_total.columns = ['customer_id', 'total_amount']
    print("고객별 총 구매 금액 (상위 10명):")
    print(customer_total.sort_values('total_amount', ascending=False).head(10))

    # 월별 평균 주문 금액
    df['month'] = df['date'].dt.month
    monthly_avg = df.groupby('month')['amount'].mean().reset_index()
    print("\n월별 평균 주문 금액:")
    print(monthly_avg)

    # 상품 카테고리별 판매량
    category_sales = df.groupby('category')['quantity'].sum().reset_index()
    print("\n카테고리별 판매량:")
    print(category_sales)

    return customer_total, monthly_avg, category_sales

if __name__ == "__main__":
    print("=== 문제 1: CSV 파일 읽기 및 기본 정보 확인 ===")
    df1 = problem1_solution()

    print("\n=== 문제 2: 결측값 처리 ===")
    df2 = problem2_solution()

    print("\n=== 문제 3: 데이터 타입 변환 ===")
    df3 = problem3_solution()

    print("\n=== 문제 4: 데이터 필터링 및 정렬 ===")
    df4 = problem4_solution()

    print("\n=== 문제 5: 그룹화 및 집계 ===")
    results = problem5_solution()