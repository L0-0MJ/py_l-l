"""
2단계: 데이터 정제 및 변환 답안 스크립트
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from datetime import datetime, timedelta

# 문제 1: 이상치 탐지 및 처리
def problem1_solution():
    """이상치 탐지 및 처리"""
    # 샘플 데이터 생성 (이상치 포함)
    np.random.seed(42)
    normal_data = np.random.normal(100, 15, 950)
    outliers = np.random.normal(200, 10, 50)
    data = np.concatenate([normal_data, outliers])

    df = pd.DataFrame({'amount': data})

    print("원본 데이터 통계:")
    print(df['amount'].describe())

    # IQR 방법으로 이상치 탐지
    Q1 = df['amount'].quantile(0.25)
    Q3 = df['amount'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    iqr_outliers = df[(df['amount'] < lower_bound) | (df['amount'] > upper_bound)]
    print(f"\nIQR 방법으로 탐지된 이상치 개수: {len(iqr_outliers)}")

    # Z-score 방법으로 이상치 탐지
    z_scores = np.abs(stats.zscore(df['amount']))
    z_outliers = df[z_scores > 3]
    print(f"Z-score 방법으로 탐지된 이상치 개수: {len(z_outliers)}")

    # 이상치를 median 값으로 대체
    df_cleaned = df.copy()
    median_value = df['amount'].median()
    df_cleaned.loc[z_scores > 3, 'amount'] = median_value

    print(f"\n처리 후 데이터 통계:")
    print(df_cleaned['amount'].describe())

    # 시각화
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.boxplot(df['amount'])
    ax1.set_title('처리 전')

    ax2.boxplot(df_cleaned['amount'])
    ax2.set_title('처리 후')

    plt.tight_layout()
    # plt.savefig('outlier_treatment.png')
    plt.show()

    return df_cleaned

# 문제 2: 데이터 정규화
def problem2_solution():
    """데이터 정규화"""
    # 샘플 데이터 생성
    np.random.seed(42)
    df = pd.DataFrame({
        'feature1': np.random.normal(100, 20, 1000),
        'feature2': np.random.exponential(2, 1000),
        'feature3': np.random.uniform(0, 1000, 1000)
    })

    print("원본 데이터 통계:")
    print(df.describe())

    # Min-Max 정규화
    minmax_scaler = MinMaxScaler()
    df_minmax = pd.DataFrame(
        minmax_scaler.fit_transform(df),
        columns=[f'{col}_minmax' for col in df.columns]
    )

    # Standard 정규화
    standard_scaler = StandardScaler()
    df_standard = pd.DataFrame(
        standard_scaler.fit_transform(df),
        columns=[f'{col}_standard' for col in df.columns]
    )

    # Robust 정규화
    robust_scaler = RobustScaler()
    df_robust = pd.DataFrame(
        robust_scaler.fit_transform(df),
        columns=[f'{col}_robust' for col in df.columns]
    )

    # 결과 비교
    result_df = pd.concat([df, df_minmax, df_standard, df_robust], axis=1)

    print("\n정규화 후 통계 (첫 번째 피처):")
    print("Min-Max:", df_minmax.iloc[:, 0].describe())
    print("Standard:", df_standard.iloc[:, 0].describe())
    print("Robust:", df_robust.iloc[:, 0].describe())

    return result_df

# 문제 3: 피처 엔지니어링
def problem3_solution():
    """피처 엔지니어링"""
    # 샘플 데이터 생성
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='H')
    df = pd.DataFrame({
        'user_id': np.random.randint(1, 1001, len(dates)),
        'login_time': dates,
        'page_views': np.random.poisson(5, len(dates)),
        'purchases': np.random.binomial(1, 0.1, len(dates))
    })

    # 날짜에서 요일, 월, 분기 추출
    df['day_of_week'] = df['login_time'].dt.dayofweek
    df['month'] = df['login_time'].dt.month
    df['quarter'] = df['login_time'].dt.quarter

    # 시간대 구분
    def categorize_time(hour):
        if 6 <= hour < 12:
            return '아침'
        elif 12 <= hour < 18:
            return '점심'
        elif 18 <= hour < 22:
            return '저녁'
        else:
            return '밤'

    df['time_category'] = df['login_time'].dt.hour.apply(categorize_time)

    # 연속된 로그인 일수 계산 (간단한 버전)
    df_sorted = df.sort_values(['user_id', 'login_time'])
    df_sorted['date'] = df_sorted['login_time'].dt.date
    user_login_streaks = df_sorted.groupby('user_id')['date'].nunique().reset_index()
    user_login_streaks.columns = ['user_id', 'login_days']

    df = df.merge(user_login_streaks, on='user_id', how='left')

    # 활동 점수 생성
    df['activity_score'] = df['login_days'] + df['page_views'] * 0.1 + df['purchases'] * 10

    print("피처 엔지니어링 결과:")
    print(df[['user_id', 'day_of_week', 'month', 'quarter',
             'time_category', 'login_days', 'activity_score']].head(10))

    return df

# 문제 4: 데이터 검증 규칙 구현
def problem4_solution():
    """데이터 검증 규칙 구현"""
    # 샘플 데이터 생성 (일부 잘못된 데이터 포함)
    df = pd.DataFrame({
        'product_id': range(1, 101),
        'price': np.random.choice([10.5, -5.0, 0, 20.0, np.nan], 100),
        'email': ['user{}@example.com'.format(i) if i % 10 != 0
                 else 'invalid_email' for i in range(1, 101)],
        'date': pd.date_range('2020-01-01', periods=100, freq='D'),
        'required_field': [f'value_{i}' if i % 15 != 0 else None for i in range(1, 101)]
    })

    validation_errors = []

    # 가격이 0보다 큰지 검증
    invalid_prices = df[df['price'] <= 0]
    if len(invalid_prices) > 0:
        validation_errors.append({
            'rule': 'price_positive',
            'count': len(invalid_prices),
            'indices': invalid_prices.index.tolist()
        })

    # 이메일 형식 검증
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    invalid_emails = df[~df['email'].str.match(email_pattern, na=False)]
    if len(invalid_emails) > 0:
        validation_errors.append({
            'rule': 'email_format',
            'count': len(invalid_emails),
            'indices': invalid_emails.index.tolist()
        })

    # 날짜 유효성 검증 (2020-2024 범위)
    min_date = pd.Timestamp('2020-01-01')
    max_date = pd.Timestamp('2024-12-31')
    invalid_dates = df[(df['date'] < min_date) | (df['date'] > max_date)]
    if len(invalid_dates) > 0:
        validation_errors.append({
            'rule': 'date_range',
            'count': len(invalid_dates),
            'indices': invalid_dates.index.tolist()
        })

    # 필수 필드 누락 검증
    missing_required = df[df['required_field'].isnull()]
    if len(missing_required) > 0:
        validation_errors.append({
            'rule': 'required_field_missing',
            'count': len(missing_required),
            'indices': missing_required.index.tolist()
        })

    # 검증 리포트 생성
    print("데이터 검증 리포트:")
    print(f"총 레코드 수: {len(df)}")
    for error in validation_errors:
        print(f"- {error['rule']}: {error['count']}개 오류")

    return validation_errors

# 문제 5: 텍스트 데이터 전처리
def problem5_solution():
    """텍스트 데이터 전처리"""
    # 샘플 리뷰 데이터 생성
    reviews = [
        "This product is AMAZING! I love it so much!!!",
        "Terrible quality. Very disappointed :(",
        "Good value for money. Recommended.",
        "NOT GOOD AT ALL! Waste of money...",
        "Excellent service and fast delivery. Happy customer!",
        "Average product, nothing special.",
        "BEST PURCHASE EVER! 5 stars!!!",
        "Poor quality, broke after one day.",
        "Great product, will buy again.",
        "Horrible experience, do not recommend."
    ]

    df = pd.DataFrame({'review': reviews})

    # 텍스트 소문자 변환
    df['review_lower'] = df['review'].str.lower()

    # 특수문자 제거
    df['review_clean'] = df['review_lower'].str.replace(r'[^\w\s]', '', regex=True)

    # 불용어 제거 (간단한 예시)
    stop_words = ['the', 'is', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
    def remove_stopwords(text):
        words = text.split()
        return ' '.join([word for word in words if word not in stop_words])

    df['review_no_stopwords'] = df['review_clean'].apply(remove_stopwords)

    # 텍스트 길이 계산
    df['text_length'] = df['review_no_stopwords'].str.len()

    # 감정 점수 계산 (간단한 키워드 기반)
    positive_words = ['amazing', 'love', 'good', 'excellent', 'great', 'best', 'happy', 'recommended']
    negative_words = ['terrible', 'disappointed', 'poor', 'horrible', 'waste', 'not good', 'broke']

    def sentiment_score(text):
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        return pos_count - neg_count

    df['sentiment_score'] = df['review'].apply(sentiment_score)

    print("텍스트 전처리 결과:")
    print(df[['review', 'review_clean', 'text_length', 'sentiment_score']].head())

    return df

if __name__ == "__main__":
    print("=== 문제 1: 이상치 탐지 및 처리 ===")
    df1 = problem1_solution()

    print("\n=== 문제 2: 데이터 정규화 ===")
    df2 = problem2_solution()

    print("\n=== 문제 3: 피처 엔지니어링 ===")
    df3 = problem3_solution()

    print("\n=== 문제 4: 데이터 검증 규칙 구현 ===")
    errors = problem4_solution()

    print("\n=== 문제 5: 텍스트 데이터 전처리 ===")
    df5 = problem5_solution()