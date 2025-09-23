"""
5단계: 데이터 품질 관리 답안 스크립트
"""

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import warnings
from typing import Dict, List, Any, Optional
import networkx as nx

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 문제 1: 데이터 품질 지표 구현
def problem1_solution():
    """데이터 품질 지표 구현"""

    class DataQualityAnalyzer:
        def __init__(self, df: pd.DataFrame):
            self.df = df
            self.quality_metrics = {}

        def calculate_completeness(self) -> Dict[str, float]:
            """완전성(Completeness) 지표 계산"""
            completeness = {}
            total_rows = len(self.df)

            for column in self.df.columns:
                non_null_count = self.df[column].notna().sum()
                completeness[column] = (non_null_count / total_rows) * 100

            overall_completeness = np.mean(list(completeness.values()))
            completeness['overall'] = overall_completeness

            logger.info(f"완전성 지표 계산 완료: 전체 {overall_completeness:.2f}%")
            return completeness

        def calculate_accuracy(self, validation_rules: Dict[str, Any]) -> Dict[str, float]:
            """정확성(Accuracy) 지표 계산"""
            accuracy = {}
            total_rows = len(self.df)

            for column, rules in validation_rules.items():
                if column not in self.df.columns:
                    continue

                valid_count = total_rows
                column_data = self.df[column].dropna()

                # 범위 검증
                if 'min_value' in rules:
                    valid_count -= (column_data < rules['min_value']).sum()
                if 'max_value' in rules:
                    valid_count -= (column_data > rules['max_value']).sum()

                # 패턴 검증
                if 'pattern' in rules and column_data.dtype == 'object':
                    valid_count -= (~column_data.str.match(rules['pattern'], na=False)).sum()

                # 허용값 검증
                if 'allowed_values' in rules:
                    valid_count -= (~column_data.isin(rules['allowed_values'])).sum()

                accuracy[column] = (valid_count / total_rows) * 100

            overall_accuracy = np.mean(list(accuracy.values())) if accuracy else 100
            accuracy['overall'] = overall_accuracy

            logger.info(f"정확성 지표 계산 완료: 전체 {overall_accuracy:.2f}%")
            return accuracy

        def calculate_consistency(self) -> Dict[str, float]:
            """일관성(Consistency) 지표 계산"""
            consistency = {}

            # 데이터 타입 일관성
            for column in self.df.columns:
                if self.df[column].dtype == 'object':
                    # 문자열 케이스 일관성
                    string_data = self.df[column].dropna().astype(str)
                    if len(string_data) > 0:
                        lowercase_count = string_data.str.islower().sum()
                        uppercase_count = string_data.str.isupper().sum()
                        mixed_case_count = len(string_data) - lowercase_count - uppercase_count

                        # 가장 일관성 있는 케이스의 비율
                        max_consistent = max(lowercase_count, uppercase_count, mixed_case_count)
                        consistency[f'{column}_case'] = (max_consistent / len(string_data)) * 100

                # 날짜 형식 일관성 (datetime 컬럼)
                elif pd.api.types.is_datetime64_any_dtype(self.df[column]):
                    # 날짜 형식이 일관된지 확인 (간단한 예시)
                    consistency[f'{column}_format'] = 100.0  # 이미 datetime이므로 일관성 있음

            overall_consistency = np.mean(list(consistency.values())) if consistency else 100
            consistency['overall'] = overall_consistency

            logger.info(f"일관성 지표 계산 완료: 전체 {overall_consistency:.2f}%")
            return consistency

        def calculate_validity(self, schema_rules: Dict[str, Any]) -> Dict[str, float]:
            """유효성(Validity) 지표 계산"""
            validity = {}
            total_rows = len(self.df)

            for column, rules in schema_rules.items():
                if column not in self.df.columns:
                    continue

                valid_count = 0
                column_data = self.df[column]

                # 데이터 타입 검증
                expected_type = rules.get('type')
                if expected_type == 'numeric':
                    valid_count = pd.to_numeric(column_data, errors='coerce').notna().sum()
                elif expected_type == 'datetime':
                    valid_count = pd.to_datetime(column_data, errors='coerce').notna().sum()
                elif expected_type == 'string':
                    valid_count = column_data.astype(str).notna().sum()
                else:
                    valid_count = column_data.notna().sum()

                validity[column] = (valid_count / total_rows) * 100

            overall_validity = np.mean(list(validity.values())) if validity else 100
            validity['overall'] = overall_validity

            logger.info(f"유효성 지표 계산 완료: 전체 {overall_validity:.2f}%")
            return validity

        def calculate_quality_score(self) -> Dict[str, float]:
            """전체 품질 점수 계산"""
            # 기본 검증 규칙 정의
            validation_rules = {
                'age': {'min_value': 0, 'max_value': 150},
                'email': {'pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'},
                'status': {'allowed_values': ['active', 'inactive', 'pending']}
            }

            schema_rules = {
                'age': {'type': 'numeric'},
                'email': {'type': 'string'},
                'created_date': {'type': 'datetime'}
            }

            # 각 지표 계산
            completeness = self.calculate_completeness()
            accuracy = self.calculate_accuracy(validation_rules)
            consistency = self.calculate_consistency()
            validity = self.calculate_validity(schema_rules)

            # 가중 평균으로 전체 점수 계산
            weights = {'completeness': 0.3, 'accuracy': 0.3, 'consistency': 0.2, 'validity': 0.2}

            overall_score = (
                completeness['overall'] * weights['completeness'] +
                accuracy['overall'] * weights['accuracy'] +
                consistency['overall'] * weights['consistency'] +
                validity['overall'] * weights['validity']
            )

            quality_metrics = {
                'completeness': completeness,
                'accuracy': accuracy,
                'consistency': consistency,
                'validity': validity,
                'overall_score': overall_score
            }

            self.quality_metrics = quality_metrics
            logger.info(f"전체 데이터 품질 점수: {overall_score:.2f}")

            return quality_metrics

        def create_quality_dashboard(self):
            """품질 점수 대시보드 생성"""
            if not self.quality_metrics:
                self.calculate_quality_score()

            # 시각화
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

            # 전체 품질 지표 비교
            overall_scores = [
                self.quality_metrics['completeness']['overall'],
                self.quality_metrics['accuracy']['overall'],
                self.quality_metrics['consistency']['overall'],
                self.quality_metrics['validity']['overall']
            ]
            labels = ['Completeness', 'Accuracy', 'Consistency', 'Validity']

            ax1.bar(labels, overall_scores, color=['#2E8B57', '#4682B4', '#DAA520', '#CD5C5C'])
            ax1.set_title('Overall Quality Metrics')
            ax1.set_ylabel('Score (%)')
            ax1.set_ylim(0, 100)

            # 컬럼별 완전성
            completeness_data = {k: v for k, v in self.quality_metrics['completeness'].items() if k != 'overall'}
            if completeness_data:
                ax2.bar(completeness_data.keys(), completeness_data.values(), color='#2E8B57')
                ax2.set_title('Completeness by Column')
                ax2.set_ylabel('Completeness (%)')
                ax2.tick_params(axis='x', rotation=45)

            # 전체 점수 게이지
            score = self.quality_metrics['overall_score']
            ax3.pie([score, 100-score], labels=[f'Quality Score\n{score:.1f}%', ''],
                   colors=['#32CD32' if score >= 80 else '#FFD700' if score >= 60 else '#FF6347', '#F0F0F0'],
                   startangle=90, counterclock=False)
            ax3.set_title('Overall Quality Score')

            # 품질 트렌드 (시뮬레이션)
            dates = pd.date_range('2023-01-01', periods=30, freq='D')
            trend_scores = [85 + np.random.normal(0, 5) for _ in range(30)]
            ax4.plot(dates, trend_scores, marker='o', color='#4682B4')
            ax4.set_title('Quality Score Trend (Simulated)')
            ax4.set_ylabel('Quality Score (%)')
            ax4.tick_params(axis='x', rotation=45)

            plt.tight_layout()
            plt.savefig('quality_dashboard.png', dpi=300, bbox_inches='tight')
            plt.show()

    # 샘플 데이터 생성 (품질 이슈 포함)
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'id': range(1, 1001),
        'name': [f'User_{i}' if i % 10 != 0 else None for i in range(1, 1001)],  # 10% 결측값
        'age': [np.random.randint(18, 80) if i % 20 != 0 else -5 for i in range(1000)],  # 일부 잘못된 값
        'email': [f'user{i}@example.com' if i % 15 != 0 else 'invalid_email' for i in range(1, 1001)],
        'status': np.random.choice(['active', 'inactive', 'pending', 'unknown'], 1000),  # 일부 잘못된 상태
        'created_date': pd.date_range('2020-01-01', periods=1000, freq='D'),
        'salary': np.random.normal(50000, 15000, 1000)
    })

    # 데이터 품질 분석 실행
    analyzer = DataQualityAnalyzer(sample_data)
    quality_metrics = analyzer.calculate_quality_score()
    analyzer.create_quality_dashboard()

    return quality_metrics

# 문제 2: 자동화된 데이터 검증
def problem2_solution():
    """자동화된 데이터 검증 (Great Expectations 스타일)"""

    class DataValidator:
        def __init__(self):
            self.expectations = []
            self.validation_results = []

        def expect_column_values_to_be_between(self, column: str, min_value: float, max_value: float):
            """컬럼 값이 특정 범위에 있는지 검증"""
            self.expectations.append({
                'type': 'range_check',
                'column': column,
                'min_value': min_value,
                'max_value': max_value
            })

        def expect_column_values_to_not_be_null(self, column: str):
            """컬럼에 null 값이 없는지 검증"""
            self.expectations.append({
                'type': 'not_null_check',
                'column': column
            })

        def expect_column_values_to_match_regex(self, column: str, regex_pattern: str):
            """컬럼 값이 정규식 패턴과 일치하는지 검증"""
            self.expectations.append({
                'type': 'regex_check',
                'column': column,
                'pattern': regex_pattern
            })

        def expect_column_values_to_be_in_set(self, column: str, value_set: List[Any]):
            """컬럼 값이 지정된 집합에 있는지 검증"""
            self.expectations.append({
                'type': 'set_check',
                'column': column,
                'allowed_values': value_set
            })

        def validate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
            """DataFrame에 대해 모든 기댓값 검증"""
            results = {
                'total_expectations': len(self.expectations),
                'passed_expectations': 0,
                'failed_expectations': 0,
                'details': []
            }

            for expectation in self.expectations:
                try:
                    result = self._validate_single_expectation(df, expectation)
                    results['details'].append(result)

                    if result['success']:
                        results['passed_expectations'] += 1
                    else:
                        results['failed_expectations'] += 1

                except Exception as e:
                    logger.error(f"검증 중 오류 발생: {e}")
                    results['details'].append({
                        'expectation': expectation,
                        'success': False,
                        'error': str(e)
                    })
                    results['failed_expectations'] += 1

            results['success_rate'] = (results['passed_expectations'] / results['total_expectations']) * 100

            logger.info(f"검증 완료: {results['passed_expectations']}/{results['total_expectations']} 통과 "
                       f"({results['success_rate']:.1f}%)")

            return results

        def _validate_single_expectation(self, df: pd.DataFrame, expectation: Dict[str, Any]) -> Dict[str, Any]:
            """단일 기댓값 검증"""
            column = expectation['column']
            exp_type = expectation['type']

            if column not in df.columns:
                return {
                    'expectation': expectation,
                    'success': False,
                    'message': f"Column '{column}' not found in DataFrame"
                }

            if exp_type == 'range_check':
                min_val = expectation['min_value']
                max_val = expectation['max_value']
                valid_count = df[(df[column] >= min_val) & (df[column] <= max_val)].shape[0]
                total_count = df[column].notna().shape[0]

                success = valid_count == total_count
                return {
                    'expectation': expectation,
                    'success': success,
                    'valid_count': valid_count,
                    'total_count': total_count,
                    'message': f"{valid_count}/{total_count} values in range [{min_val}, {max_val}]"
                }

            elif exp_type == 'not_null_check':
                null_count = df[column].isnull().sum()
                success = null_count == 0
                return {
                    'expectation': expectation,
                    'success': success,
                    'null_count': null_count,
                    'message': f"{null_count} null values found" if not success else "No null values"
                }

            elif exp_type == 'regex_check':
                pattern = expectation['pattern']
                string_data = df[column].astype(str)
                valid_count = string_data.str.match(pattern, na=False).sum()
                total_count = len(string_data)

                success = valid_count == total_count
                return {
                    'expectation': expectation,
                    'success': success,
                    'valid_count': valid_count,
                    'total_count': total_count,
                    'message': f"{valid_count}/{total_count} values match pattern"
                }

            elif exp_type == 'set_check':
                allowed_values = expectation['allowed_values']
                valid_count = df[column].isin(allowed_values).sum()
                total_count = df[column].notna().shape[0]

                success = valid_count == total_count
                return {
                    'expectation': expectation,
                    'success': success,
                    'valid_count': valid_count,
                    'total_count': total_count,
                    'message': f"{valid_count}/{total_count} values in allowed set"
                }

        def generate_validation_report(self, results: Dict[str, Any], output_file: str = 'validation_report.html'):
            """검증 결과 HTML 리포트 생성"""
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Data Validation Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                    .success {{ color: green; }}
                    .failure {{ color: red; }}
                    .expectation {{ margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 3px; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Data Validation Report</h1>
                    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p>Total Expectations: {results['total_expectations']}</p>
                    <p class="{'success' if results['success_rate'] >= 80 else 'failure'}">
                        Success Rate: {results['success_rate']:.1f}%
                    </p>
                </div>

                <h2>Validation Results</h2>
                <table>
                    <tr>
                        <th>Expectation Type</th>
                        <th>Column</th>
                        <th>Status</th>
                        <th>Message</th>
                    </tr>
            """

            for detail in results['details']:
                exp = detail['expectation']
                status_class = 'success' if detail['success'] else 'failure'
                status_text = 'PASS' if detail['success'] else 'FAIL'

                html_content += f"""
                    <tr>
                        <td>{exp['type']}</td>
                        <td>{exp['column']}</td>
                        <td class="{status_class}">{status_text}</td>
                        <td>{detail.get('message', '')}</td>
                    </tr>
                """

            html_content += """
                </table>
            </body>
            </html>
            """

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)

            logger.info(f"검증 리포트 생성: {output_file}")

    # 샘플 데이터 및 검증 실행
    sample_data = pd.DataFrame({
        'user_id': range(1, 101),
        'age': np.random.randint(18, 80, 100),
        'email': [f'user{i}@example.com' for i in range(1, 101)],
        'status': np.random.choice(['active', 'inactive'], 100),
        'score': np.random.uniform(0, 100, 100)
    })

    # 일부 데이터 품질 이슈 주입
    sample_data.loc[5, 'age'] = 200  # 범위 초과
    sample_data.loc[10, 'email'] = 'invalid_email'  # 잘못된 이메일
    sample_data.loc[15, 'status'] = 'unknown'  # 허용되지 않는 값

    # 검증 실행
    validator = DataValidator()
    validator.expect_column_values_to_be_between('age', 0, 120)
    validator.expect_column_values_to_not_be_null('user_id')
    validator.expect_column_values_to_match_regex('email', r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    validator.expect_column_values_to_be_in_set('status', ['active', 'inactive'])
    validator.expect_column_values_to_be_between('score', 0, 100)

    validation_results = validator.validate_dataframe(sample_data)
    validator.generate_validation_report(validation_results)

    return validation_results

# 문제 3: 데이터 프로파일링
def problem3_solution():
    """데이터 프로파일링"""

    class DataProfiler:
        def __init__(self, df: pd.DataFrame):
            self.df = df
            self.profile_results = {}

        def analyze_distributions(self) -> Dict[str, Any]:
            """데이터 분포 분석"""
            distributions = {}

            for column in self.df.columns:
                col_data = self.df[column].dropna()

                if pd.api.types.is_numeric_dtype(col_data):
                    distributions[column] = {
                        'type': 'numeric',
                        'mean': float(col_data.mean()),
                        'std': float(col_data.std()),
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        'quartiles': {
                            'q1': float(col_data.quantile(0.25)),
                            'q2': float(col_data.quantile(0.50)),
                            'q3': float(col_data.quantile(0.75))
                        },
                        'skewness': float(col_data.skew()),
                        'kurtosis': float(col_data.kurtosis())
                    }
                elif pd.api.types.is_categorical_dtype(col_data) or col_data.dtype == 'object':
                    value_counts = col_data.value_counts()
                    distributions[column] = {
                        'type': 'categorical',
                        'unique_count': len(value_counts),
                        'most_frequent': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                        'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                        'value_counts': value_counts.head(10).to_dict()
                    }

            logger.info(f"분포 분석 완료: {len(distributions)}개 컬럼")
            return distributions

        def analyze_correlations(self) -> Dict[str, Any]:
            """상관관계 분석"""
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns

            if len(numeric_cols) < 2:
                logger.warning("상관관계 분석을 위한 수치형 컬럼이 부족합니다")
                return {}

            correlation_matrix = self.df[numeric_cols].corr()

            # 강한 상관관계 찾기 (절댓값 0.7 이상)
            strong_correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) >= 0.7:
                        strong_correlations.append({
                            'column1': correlation_matrix.columns[i],
                            'column2': correlation_matrix.columns[j],
                            'correlation': float(corr_value)
                        })

            correlations = {
                'correlation_matrix': correlation_matrix.to_dict(),
                'strong_correlations': strong_correlations,
                'numeric_columns': list(numeric_cols)
            }

            logger.info(f"상관관계 분석 완료: {len(strong_correlations)}개 강한 상관관계 발견")
            return correlations

        def detect_outlier_patterns(self) -> Dict[str, Any]:
            """이상치 패턴 탐지"""
            outlier_patterns = {}

            numeric_cols = self.df.select_dtypes(include=[np.number]).columns

            for column in numeric_cols:
                col_data = self.df[column].dropna()

                if len(col_data) == 0:
                    continue

                # IQR 방법
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                iqr_outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]

                # Z-score 방법
                z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
                zscore_outliers = col_data[z_scores > 3]

                outlier_patterns[column] = {
                    'iqr_outliers': {
                        'count': len(iqr_outliers),
                        'percentage': (len(iqr_outliers) / len(col_data)) * 100,
                        'values': iqr_outliers.tolist()[:10]  # 최대 10개만 저장
                    },
                    'zscore_outliers': {
                        'count': len(zscore_outliers),
                        'percentage': (len(zscore_outliers) / len(col_data)) * 100,
                        'values': zscore_outliers.tolist()[:10]
                    },
                    'bounds': {
                        'iqr_lower': float(lower_bound),
                        'iqr_upper': float(upper_bound)
                    }
                }

            logger.info(f"이상치 패턴 분석 완료: {len(outlier_patterns)}개 수치형 컬럼")
            return outlier_patterns

        def generate_profile_report(self) -> Dict[str, Any]:
            """종합 프로파일링 리포트 생성"""
            logger.info("데이터 프로파일링 시작")

            # 기본 정보
            basic_info = {
                'shape': self.df.shape,
                'columns': list(self.df.columns),
                'dtypes': self.df.dtypes.astype(str).to_dict(),
                'missing_values': self.df.isnull().sum().to_dict(),
                'memory_usage': float(self.df.memory_usage(deep=True).sum() / 1024**2)  # MB
            }

            # 각종 분석 수행
            distributions = self.analyze_distributions()
            correlations = self.analyze_correlations()
            outlier_patterns = self.detect_outlier_patterns()

            profile_report = {
                'timestamp': datetime.now().isoformat(),
                'basic_info': basic_info,
                'distributions': distributions,
                'correlations': correlations,
                'outlier_patterns': outlier_patterns
            }

            self.profile_results = profile_report

            # JSON 리포트 저장
            with open('data_profile_report.json', 'w', encoding='utf-8') as f:
                json.dump(profile_report, f, indent=2, ensure_ascii=False)

            logger.info("프로파일링 리포트 생성 완료: data_profile_report.json")
            return profile_report

        def create_profile_visualizations(self):
            """프로파일링 시각화"""
            if not self.profile_results:
                self.generate_profile_report()

            fig, axes = plt.subplots(2, 2, figsize=(16, 12))

            # 결측값 비율
            missing_data = self.profile_results['basic_info']['missing_values']
            missing_df = pd.DataFrame(list(missing_data.items()), columns=['Column', 'Missing_Count'])
            missing_df['Missing_Percentage'] = (missing_df['Missing_Count'] / self.df.shape[0]) * 100

            axes[0, 0].bar(missing_df['Column'], missing_df['Missing_Percentage'])
            axes[0, 0].set_title('Missing Values by Column')
            axes[0, 0].set_ylabel('Missing Percentage (%)')
            axes[0, 0].tick_params(axis='x', rotation=45)

            # 상관관계 히트맵
            if self.profile_results['correlations']:
                corr_matrix = pd.DataFrame(self.profile_results['correlations']['correlation_matrix'])
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[0, 1])
                axes[0, 1].set_title('Correlation Heatmap')

            # 이상치 개수
            outlier_counts = []
            columns = []
            for col, patterns in self.profile_results['outlier_patterns'].items():
                columns.append(col)
                outlier_counts.append(patterns['iqr_outliers']['count'])

            if outlier_counts:
                axes[1, 0].bar(columns, outlier_counts, color='red', alpha=0.7)
                axes[1, 0].set_title('Outlier Count by Column (IQR Method)')
                axes[1, 0].set_ylabel('Outlier Count')
                axes[1, 0].tick_params(axis='x', rotation=45)

            # 데이터 타입 분포
            dtype_counts = pd.Series(self.profile_results['basic_info']['dtypes']).value_counts()
            axes[1, 1].pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
            axes[1, 1].set_title('Data Type Distribution')

            plt.tight_layout()
            plt.savefig('profile_visualizations.png', dpi=300, bbox_inches='tight')
            plt.show()

    # 샘플 데이터 생성 및 프로파일링
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'user_id': range(1, 1001),
        'age': np.random.normal(35, 12, 1000),
        'income': np.random.normal(50000, 15000, 1000),
        'score': np.random.beta(2, 5, 1000) * 100,
        'category': np.random.choice(['A', 'B', 'C', 'D'], 1000),
        'is_active': np.random.choice([True, False], 1000),
        'registration_date': pd.date_range('2020-01-01', periods=1000, freq='D')
    })

    # 일부 이상치 및 결측값 주입
    sample_data.loc[np.random.choice(1000, 50, replace=False), 'age'] = np.random.normal(80, 5, 50)
    sample_data.loc[np.random.choice(1000, 30, replace=False), 'income'] = None

    profiler = DataProfiler(sample_data)
    profile_report = profiler.generate_profile_report()
    profiler.create_profile_visualizations()

    return profile_report

# 문제 4: 로깅 및 모니터링 시스템
def problem4_solution():
    """로깅 및 모니터링 시스템"""

    class DataProcessingMonitor:
        def __init__(self):
            self.setup_logging()
            self.metrics = {
                'processing_times': [],
                'error_counts': {},
                'data_volumes': [],
                'quality_scores': []
            }

        def setup_logging(self):
            """구조화된 로깅 설정"""
            # JSON 포맷터
            class JSONFormatter(logging.Formatter):
                def format(self, record):
                    log_entry = {
                        'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                        'level': record.levelname,
                        'logger': record.name,
                        'message': record.getMessage(),
                        'module': record.module,
                        'function': record.funcName,
                        'line': record.lineno
                    }

                    # 추가 속성이 있으면 포함
                    if hasattr(record, 'user_id'):
                        log_entry['user_id'] = record.user_id
                    if hasattr(record, 'processing_time'):
                        log_entry['processing_time'] = record.processing_time
                    if hasattr(record, 'data_size'):
                        log_entry['data_size'] = record.data_size

                    return json.dumps(log_entry, ensure_ascii=False)

            # 로거 설정
            self.logger = logging.getLogger('DataProcessingMonitor')
            self.logger.setLevel(logging.INFO)

            # 파일 핸들러
            file_handler = logging.FileHandler('data_processing.log')
            file_handler.setFormatter(JSONFormatter())
            self.logger.addHandler(file_handler)

            # 메트릭 로거
            self.metrics_logger = logging.getLogger('MetricsLogger')
            metrics_handler = logging.FileHandler('processing_metrics.log')
            metrics_handler.setFormatter(JSONFormatter())
            self.metrics_logger.addHandler(metrics_handler)

        def log_processing_start(self, process_name: str, data_size: int, **kwargs):
            """처리 시작 로깅"""
            self.logger.info(f"Processing started: {process_name}",
                           extra={'data_size': data_size, 'process_name': process_name, **kwargs})

        def log_processing_end(self, process_name: str, processing_time: float, success: bool, **kwargs):
            """처리 완료 로깅"""
            level = logging.INFO if success else logging.ERROR
            status = "completed" if success else "failed"

            self.logger.log(level, f"Processing {status}: {process_name}",
                          extra={'processing_time': processing_time, 'process_name': process_name, **kwargs})

            # 메트릭 수집
            self.metrics['processing_times'].append({
                'process_name': process_name,
                'processing_time': processing_time,
                'success': success,
                'timestamp': datetime.now().isoformat()
            })

        def log_error(self, error_type: str, error_message: str, **kwargs):
            """에러 로깅 및 분류"""
            self.logger.error(f"Error occurred: {error_type} - {error_message}",
                            extra={'error_type': error_type, **kwargs})

            # 에러 카운트 업데이트
            if error_type not in self.metrics['error_counts']:
                self.metrics['error_counts'][error_type] = 0
            self.metrics['error_counts'][error_type] += 1

        def collect_performance_metrics(self, data_volume: int, quality_score: float):
            """성능 메트릭 수집"""
            self.metrics['data_volumes'].append({
                'volume': data_volume,
                'timestamp': datetime.now().isoformat()
            })

            self.metrics['quality_scores'].append({
                'score': quality_score,
                'timestamp': datetime.now().isoformat()
            })

            self.metrics_logger.info("Performance metrics collected",
                                   extra={'data_volume': data_volume, 'quality_score': quality_score})

        def generate_monitoring_dashboard(self):
            """모니터링 대시보드 생성"""
            # 처리 시간 통계
            if self.metrics['processing_times']:
                processing_df = pd.DataFrame(self.metrics['processing_times'])
                avg_processing_time = processing_df['processing_time'].mean()
                success_rate = (processing_df['success'].sum() / len(processing_df)) * 100
            else:
                avg_processing_time = 0
                success_rate = 0

            # 시각화
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

            # 처리 시간 트렌드
            if self.metrics['processing_times']:
                times_df = pd.DataFrame(self.metrics['processing_times'])
                times_df['timestamp'] = pd.to_datetime(times_df['timestamp'])
                ax1.plot(times_df['timestamp'], times_df['processing_time'], marker='o')
                ax1.set_title('Processing Time Trend')
                ax1.set_ylabel('Processing Time (seconds)')
                ax1.tick_params(axis='x', rotation=45)

            # 에러 분포
            if self.metrics['error_counts']:
                error_types = list(self.metrics['error_counts'].keys())
                error_counts = list(self.metrics['error_counts'].values())
                ax2.pie(error_counts, labels=error_types, autopct='%1.1f%%')
                ax2.set_title('Error Distribution')

            # 데이터 볼륨 트렌드
            if self.metrics['data_volumes']:
                volume_df = pd.DataFrame(self.metrics['data_volumes'])
                volume_df['timestamp'] = pd.to_datetime(volume_df['timestamp'])
                ax3.plot(volume_df['timestamp'], volume_df['volume'], marker='s', color='green')
                ax3.set_title('Data Volume Trend')
                ax3.set_ylabel('Data Volume')
                ax3.tick_params(axis='x', rotation=45)

            # 품질 점수 트렌드
            if self.metrics['quality_scores']:
                quality_df = pd.DataFrame(self.metrics['quality_scores'])
                quality_df['timestamp'] = pd.to_datetime(quality_df['timestamp'])
                ax4.plot(quality_df['timestamp'], quality_df['score'], marker='d', color='orange')
                ax4.set_title('Quality Score Trend')
                ax4.set_ylabel('Quality Score')
                ax4.set_ylim(0, 100)
                ax4.tick_params(axis='x', rotation=45)

            plt.tight_layout()
            plt.savefig('monitoring_dashboard.png', dpi=300, bbox_inches='tight')
            plt.show()

            # 요약 통계
            dashboard_summary = {
                'avg_processing_time': avg_processing_time,
                'success_rate': success_rate,
                'total_errors': sum(self.metrics['error_counts'].values()),
                'error_types': len(self.metrics['error_counts']),
                'dashboard_generated': datetime.now().isoformat()
            }

            return dashboard_summary

    # 모니터링 시스템 테스트
    monitor = DataProcessingMonitor()

    # 시뮬레이션된 처리 작업들
    processes = [
        {'name': 'data_ingestion', 'duration': 2.5, 'data_size': 10000, 'success': True},
        {'name': 'data_cleaning', 'duration': 5.2, 'data_size': 9500, 'success': True},
        {'name': 'data_validation', 'duration': 1.8, 'data_size': 9500, 'success': False},
        {'name': 'data_transformation', 'duration': 3.7, 'data_size': 9200, 'success': True},
        {'name': 'data_export', 'duration': 2.1, 'data_size': 9200, 'success': True}
    ]

    # 처리 시뮬레이션
    for process in processes:
        monitor.log_processing_start(process['name'], process['data_size'])

        if not process['success']:
            monitor.log_error('validation_error', 'Data validation failed due to schema mismatch',
                            process_name=process['name'])

        monitor.log_processing_end(process['name'], process['duration'], process['success'])
        monitor.collect_performance_metrics(process['data_size'], np.random.uniform(70, 95))

    # 대시보드 생성
    dashboard_summary = monitor.generate_monitoring_dashboard()

    return dashboard_summary

# 문제 5: 데이터 리니지 추적
def problem5_solution():
    """데이터 리니지 추적"""

    class DataLineageTracker:
        def __init__(self):
            self.lineage_graph = nx.DiGraph()
            self.transformations = []
            self.metadata = {}

        def add_data_source(self, source_id: str, source_type: str, location: str, **metadata):
            """데이터 소스 추가"""
            self.lineage_graph.add_node(source_id,
                                      node_type='source',
                                      source_type=source_type,
                                      location=location,
                                      created_at=datetime.now().isoformat(),
                                      **metadata)

            self.metadata[source_id] = {
                'type': 'source',
                'source_type': source_type,
                'location': location,
                'metadata': metadata
            }

        def add_transformation(self, transform_id: str, input_datasets: List[str],
                             output_dataset: str, transformation_type: str,
                             transformation_logic: str, **metadata):
            """데이터 변환 추가"""

            # 변환 노드 추가
            self.lineage_graph.add_node(transform_id,
                                      node_type='transformation',
                                      transformation_type=transformation_type,
                                      logic=transformation_logic,
                                      created_at=datetime.now().isoformat(),
                                      **metadata)

            # 출력 데이터셋 노드 추가
            self.lineage_graph.add_node(output_dataset,
                                      node_type='dataset',
                                      created_at=datetime.now().isoformat())

            # 입력 -> 변환 엣지 추가
            for input_dataset in input_datasets:
                self.lineage_graph.add_edge(input_dataset, transform_id)

            # 변환 -> 출력 엣지 추가
            self.lineage_graph.add_edge(transform_id, output_dataset)

            # 변환 기록
            transformation_record = {
                'transform_id': transform_id,
                'input_datasets': input_datasets,
                'output_dataset': output_dataset,
                'transformation_type': transformation_type,
                'transformation_logic': transformation_logic,
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata
            }

            self.transformations.append(transformation_record)

        def get_upstream_lineage(self, dataset_id: str, max_depth: int = 10) -> List[str]:
            """데이터셋의 상위 리니지 추적"""
            upstream_nodes = []

            def traverse_upstream(node, depth=0):
                if depth >= max_depth:
                    return

                predecessors = list(self.lineage_graph.predecessors(node))
                for pred in predecessors:
                    if pred not in upstream_nodes:
                        upstream_nodes.append(pred)
                        traverse_upstream(pred, depth + 1)

            traverse_upstream(dataset_id)
            return upstream_nodes

        def get_downstream_lineage(self, dataset_id: str, max_depth: int = 10) -> List[str]:
            """데이터셋의 하위 리니지 추적"""
            downstream_nodes = []

            def traverse_downstream(node, depth=0):
                if depth >= max_depth:
                    return

                successors = list(self.lineage_graph.successors(node))
                for succ in successors:
                    if succ not in downstream_nodes:
                        downstream_nodes.append(succ)
                        traverse_downstream(succ, depth + 1)

            traverse_downstream(dataset_id)
            return downstream_nodes

        def analyze_impact(self, dataset_id: str) -> Dict[str, Any]:
            """데이터 변경 영향도 분석"""
            downstream_datasets = self.get_downstream_lineage(dataset_id)

            # 영향받는 변환 및 데이터셋 분류
            affected_transformations = []
            affected_datasets = []

            for node in downstream_datasets:
                node_data = self.lineage_graph.nodes[node]
                if node_data.get('node_type') == 'transformation':
                    affected_transformations.append(node)
                elif node_data.get('node_type') == 'dataset':
                    affected_datasets.append(node)

            impact_analysis = {
                'source_dataset': dataset_id,
                'total_affected_nodes': len(downstream_datasets),
                'affected_transformations': affected_transformations,
                'affected_datasets': affected_datasets,
                'impact_severity': self._calculate_impact_severity(len(affected_datasets)),
                'analysis_timestamp': datetime.now().isoformat()
            }

            return impact_analysis

        def _calculate_impact_severity(self, affected_count: int) -> str:
            """영향도 심각도 계산"""
            if affected_count == 0:
                return 'none'
            elif affected_count <= 2:
                return 'low'
            elif affected_count <= 5:
                return 'medium'
            else:
                return 'high'

        def track_schema_change(self, dataset_id: str, old_schema: Dict, new_schema: Dict):
            """스키마 변경 추적"""
            change_record = {
                'dataset_id': dataset_id,
                'change_type': 'schema_change',
                'old_schema': old_schema,
                'new_schema': new_schema,
                'timestamp': datetime.now().isoformat(),
                'changes': self._compare_schemas(old_schema, new_schema)
            }

            # 변경 이력에 추가
            if 'change_history' not in self.metadata:
                self.metadata['change_history'] = []
            self.metadata['change_history'].append(change_record)

            return change_record

        def _compare_schemas(self, old_schema: Dict, new_schema: Dict) -> Dict[str, List]:
            """스키마 비교"""
            old_columns = set(old_schema.get('columns', []))
            new_columns = set(new_schema.get('columns', []))

            changes = {
                'added_columns': list(new_columns - old_columns),
                'removed_columns': list(old_columns - new_columns),
                'type_changes': []
            }

            # 타입 변경 확인
            for column in old_columns & new_columns:
                old_type = old_schema.get('types', {}).get(column)
                new_type = new_schema.get('types', {}).get(column)
                if old_type and new_type and old_type != new_type:
                    changes['type_changes'].append({
                        'column': column,
                        'old_type': old_type,
                        'new_type': new_type
                    })

            return changes

        def visualize_lineage(self, focus_dataset: str = None):
            """리니지 시각화"""
            plt.figure(figsize=(16, 12))

            # 노드 위치 계산 (계층적 레이아웃)
            pos = nx.spring_layout(self.lineage_graph, k=3, iterations=50)

            # 노드 타입별 색상
            node_colors = []
            node_sizes = []
            for node in self.lineage_graph.nodes():
                node_data = self.lineage_graph.nodes[node]
                node_type = node_data.get('node_type', 'unknown')

                if node_type == 'source':
                    node_colors.append('lightblue')
                    node_sizes.append(800)
                elif node_type == 'transformation':
                    node_colors.append('lightcoral')
                    node_sizes.append(600)
                elif node_type == 'dataset':
                    node_colors.append('lightgreen')
                    node_sizes.append(500)
                else:
                    node_colors.append('gray')
                    node_sizes.append(400)

            # 그래프 그리기
            nx.draw(self.lineage_graph, pos,
                   node_color=node_colors,
                   node_size=node_sizes,
                   with_labels=True,
                   font_size=8,
                   font_weight='bold',
                   arrows=True,
                   arrowsize=20,
                   edge_color='gray',
                   alpha=0.7)

            # 포커스 데이터셋 강조
            if focus_dataset and focus_dataset in self.lineage_graph.nodes():
                nx.draw_networkx_nodes(self.lineage_graph, pos,
                                     nodelist=[focus_dataset],
                                     node_color='yellow',
                                     node_size=1000,
                                     alpha=0.8)

            plt.title('Data Lineage Graph')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig('data_lineage.png', dpi=300, bbox_inches='tight')
            plt.show()

        def export_lineage_metadata(self, output_file: str = 'lineage_metadata.json'):
            """리니지 메타데이터 내보내기"""
            export_data = {
                'nodes': dict(self.lineage_graph.nodes(data=True)),
                'edges': list(self.lineage_graph.edges()),
                'transformations': self.transformations,
                'metadata': self.metadata,
                'export_timestamp': datetime.now().isoformat()
            }

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            logger.info(f"리니지 메타데이터 내보내기 완료: {output_file}")
            return export_data

    # 데이터 리니지 추적 시뮬레이션
    tracker = DataLineageTracker()

    # 데이터 소스 추가
    tracker.add_data_source('raw_sales_data', 'csv', '/data/raw/sales.csv',
                          schema={'columns': ['order_id', 'customer_id', 'amount', 'date']})
    tracker.add_data_source('customer_data', 'database', 'postgres://db/customers',
                          schema={'columns': ['customer_id', 'name', 'email', 'region']})

    # 변환 과정 추가
    tracker.add_transformation('clean_sales',
                             ['raw_sales_data'],
                             'cleaned_sales',
                             'data_cleaning',
                             'Remove nulls, standardize date format')

    tracker.add_transformation('join_customer_sales',
                             ['cleaned_sales', 'customer_data'],
                             'sales_with_customer',
                             'join',
                             'JOIN sales ON customer_id')

    tracker.add_transformation('monthly_aggregation',
                             ['sales_with_customer'],
                             'monthly_sales_summary',
                             'aggregation',
                             'GROUP BY month, region SUM(amount)')

    # 영향도 분석
    impact = tracker.analyze_impact('raw_sales_data')
    print("영향도 분석 결과:")
    print(f"- 영향받는 변환: {len(impact['affected_transformations'])}개")
    print(f"- 영향받는 데이터셋: {len(impact['affected_datasets'])}개")
    print(f"- 심각도: {impact['impact_severity']}")

    # 스키마 변경 추적
    old_schema = {'columns': ['order_id', 'customer_id', 'amount', 'date'],
                 'types': {'order_id': 'int', 'customer_id': 'int', 'amount': 'float', 'date': 'date'}}
    new_schema = {'columns': ['order_id', 'customer_id', 'amount', 'date', 'product_id'],
                 'types': {'order_id': 'int', 'customer_id': 'int', 'amount': 'decimal', 'date': 'datetime', 'product_id': 'int'}}

    schema_change = tracker.track_schema_change('raw_sales_data', old_schema, new_schema)
    print(f"\n스키마 변경 사항:")
    print(f"- 추가된 컬럼: {schema_change['changes']['added_columns']}")
    print(f"- 타입 변경: {len(schema_change['changes']['type_changes'])}개")

    # 리니지 시각화
    tracker.visualize_lineage('monthly_sales_summary')

    # 메타데이터 내보내기
    export_data = tracker.export_lineage_metadata()

    return {
        'impact_analysis': impact,
        'schema_change': schema_change,
        'total_nodes': len(tracker.lineage_graph.nodes()),
        'total_edges': len(tracker.lineage_graph.edges())
    }

if __name__ == "__main__":
    print("=== 문제 1: 데이터 품질 지표 구현 ===")
    quality_results = problem1_solution()
    print(f"전체 품질 점수: {quality_results['overall_score']:.2f}")

    print("\n=== 문제 2: 자동화된 데이터 검증 ===")
    validation_results = problem2_solution()
    print(f"검증 성공률: {validation_results['success_rate']:.1f}%")

    print("\n=== 문제 3: 데이터 프로파일링 ===")
    profile_results = problem3_solution()
    print(f"분석된 컬럼 수: {len(profile_results['distributions'])}")

    print("\n=== 문제 4: 로깅 및 모니터링 시스템 ===")
    monitoring_results = problem4_solution()
    print(f"평균 처리 시간: {monitoring_results['avg_processing_time']:.2f}초")

    print("\n=== 문제 5: 데이터 리니지 추적 ===")
    lineage_results = problem5_solution()
    print(f"리니지 노드 수: {lineage_results['total_nodes']}")
    print(f"리니지 엣지 수: {lineage_results['total_edges']}")