# 🐍 1단계 Python 기초 강화 실습 문제

"""
데이터 사이언스를 위한 Python 기초 마스터하기
각 문제를 순서대로 해결하면서 핵심 개념을 체득하세요!
"""

# ==========================================
# 문제 1: 클래스와 속성 관리 (20분)
# ==========================================

"""
📝 문제 1: CSV 데이터 읽기 클래스 만들기

요구사항:
1. CSVReader 클래스 구현
2. 파일 경로를 받아서 초기화
3. @property 데코레이터로 데이터 접근
4. 파일이 없으면 FileNotFoundError 발생
5. 데이터 기본 정보 출력 메서드 구현

힌트: 
- __init__에서 파일 존재 확인
- @property로 데이터를 지연 로딩
- pandas 없이 기본 csv 모듈 사용
"""

import csv
import os

class CSVReader:
    def __init__(self, file_path):
        # TODO: 파일 경로 저장 및 존재 확인
        pass
    
    @property 
    def data(self):
        # TODO: CSV 데이터를 리스트로 반환 (헤더 포함)
        pass
    
    @property
    def headers(self):
        # TODO: 첫 번째 행(헤더) 반환
        pass
    
    def get_info(self):
        # TODO: 행수, 열수, 헤더 정보 출력
        pass

# 테스트 코드 작성해보세요:
# reader = CSVReader("test.csv")
# print(reader.headers)
# reader.get_info()

print("=" * 50)

# ==========================================
# 문제 2: 데코레이터 만들기 (25분) 
# ==========================================

"""
📝 문제 2: 실행 시간 측정 데코레이터

요구사항:
1. 함수 실행 시간을 측정하는 데코레이터
2. 함수 이름, 실행 시간, 결과를 출력
3. 에러 발생 시에도 시간 측정
4. 여러 함수에 적용 가능하도록 범용적으로 구현

힌트:
- functools.wraps 사용
- time.time()으로 시간 측정
- try-except로 에러 처리
"""

import time
import functools

def measure_time(func):
    # TODO: 실행 시간 측정 데코레이터 구현
    pass

# 테스트용 함수들 - 데코레이터 적용해보세요
@measure_time
def slow_calculation(n):
    """느린 계산 함수"""
    result = 0
    for i in range(n):
        result += i ** 2
    return result

@measure_time 
def file_processing(data_size):
    """파일 처리 시뮬레이션"""
    # 처리 시간 시뮬레이션
    time.sleep(data_size * 0.001)
    return f"Processed {data_size} records"

# 테스트 실행해보세요:
# slow_calculation(100000)
# file_processing(1000)

print("=" * 50)

# ==========================================
# 문제 3: 파일 처리와 예외 처리 (30분)
# ==========================================

"""
📝 문제 3: 데이터 처리 클래스 + 컨텍스트 매니저

요구사항:
1. DataProcessor 클래스 구현
2. 안전한 파일 읽기/쓰기 기능
3. 커스텀 예외 클래스 3개 이상
4. with 문으로 사용 가능한 컨텍스트 매니저
5. 처리 중 에러 로그를 파일에 기록

힌트:
- __enter__, __exit__ 메서드 구현
- 여러 종류의 Exception 상속 클래스
- logging 모듈 사용 고려
"""

# 1단계: 커스텀 예외 클래스들 정의
class DataProcessingError(Exception):
    # TODO: 데이터 처리 관련 예외
    pass

class InvalidDataFormatError(Exception):  
    # TODO: 잘못된 데이터 형식 예외
    pass

class ProcessingTimeoutError(Exception):
    # TODO: 처리 시간 초과 예외  
    pass

# 2단계: 메인 데이터 프로세서 클래스
class DataProcessor:
    def __init__(self, input_file, output_file, timeout=30):
        # TODO: 초기화 - 파일 경로, 타임아웃 설정
        pass
    
    def __enter__(self):
        # TODO: 컨텍스트 매니저 진입
        # 로그 파일 열기, 처리 시작 기록
        pass
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # TODO: 컨텍스트 매니저 종료
        # 에러 로깅, 파일 정리
        pass
    
    def process_line(self, line):
        # TODO: 한 줄씩 데이터 처리
        # 빈 줄이면 InvalidDataFormatError
        # 특정 형식 검증
        pass
    
    def process_file(self):
        # TODO: 전체 파일 처리
        # 각 줄을 process_line으로 처리
        # 결과를 output_file에 저장
        pass

# 사용 예시:
# with DataProcessor("input.txt", "output.txt") as processor:
#     processor.process_file()

print("=" * 50)

# ==========================================
# 보너스 문제: 종합 응용 (15분)
# ==========================================

"""
📝 보너스: 위 3개 기능을 조합한 CSV 분석기

요구사항:
1. CSVAnalyzer 클래스 - 위 3개 기능 모두 활용
2. @measure_time 데코레이터로 성능 측정
3. 컨텍스트 매니저로 안전한 파일 처리  
4. 기본 통계(평균, 최대값 등) 계산
5. 에러 발생시 로그 파일에 기록
"""

class CSVAnalyzer:
    def __init__(self, csv_file):
        # TODO: CSV 파일 경로 저장
        pass
    
    def __enter__(self):
        # TODO: 컨텍스트 매니저 진입
        pass
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # TODO: 컨텍스트 매니저 종료
        pass
    
    @measure_time
    def analyze_numeric_column(self, column_name):
        # TODO: 숫자 컬럼 분석 (평균, 최대, 최소값)
        pass
    
    @measure_time  
    def get_summary_report(self):
        # TODO: 전체 데이터 요약 리포트 생성
        pass

# 사용 예시:
# with CSVAnalyzer("sales_data.csv") as analyzer:
#     analyzer.analyze_numeric_column("price")
#     report = analyzer.get_summary_report()

print("=" * 50)

"""
🎯 도전 과제:
1. 실제 CSV 파일 만들어서 테스트
2. 각 클래스에 __repr__ 메서드 추가  
3. 타입 힌트(Type Hints) 추가
4. docstring 추가
5. 단위 테스트 코드 작성

완료되면 다음으로 넘어갑니다! 💪
"""