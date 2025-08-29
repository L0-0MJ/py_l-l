# 🎯 1단계 실습 문제 힌트 가이드

## 📝 **문제 1: CSVReader 클래스 힌트**

### 단계별 구현 가이드

#### 1️⃣ __init__ 메서드
```python
def __init__(self, file_path):
    # 1. 파일 경로 저장
    self.file_path = file_path
    
    # 2. 파일 존재 확인
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
    
    # 3. 데이터 캐시 초기화 (첫 읽기 시에만 로드)
    self._data = None
```

#### 2️⃣ @property data
```python
@property
def data(self):
    # 지연 로딩: 처음 접근할 때만 파일 읽기
    if self._data is None:
        with open(self.file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            self._data = list(reader)  # 모든 행을 리스트로 저장
    return self._data
```

#### 3️⃣ @property headers
```python
@property
def headers(self):
    # 첫 번째 행이 헤더
    return self.data[0] if self.data else []
```

#### 4️⃣ get_info 메서드
```python
def get_info(self):
    print(f"📊 CSV 파일 정보:")
    print(f"   파일: {self.file_path}")
    print(f"   총 행수: {len(self.data)}")
    print(f"   열수: {len(self.headers)}")
    print(f"   헤더: {', '.join(self.headers)}")
```

---

## ⏱️ **문제 2: measure_time 데코레이터 힌트**

### 단계별 구현 가이드

#### 1️⃣ 기본 구조
```python
def measure_time(func):
    @functools.wraps(func)  # 원본 함수 정보 보존
    def wrapper(*args, **kwargs):
        # 시작 시간 기록
        start_time = time.time()
        
        try:
            # 함수 실행
            result = func(*args, **kwargs)
            
            # 성공시 처리
            end_time = time.time()
            execution_time = end_time - start_time
            
            print(f"✅ [{func.__name__}] 실행 완료:")
            print(f"   ⏱️ 실행시간: {execution_time:.4f}초")
            print(f"   📤 결과: {result}")
            
            return result
            
        except Exception as e:
            # 에러시 처리
            end_time = time.time()
            execution_time = end_time - start_time
            
            print(f"❌ [{func.__name__}] 실행 실패:")
            print(f"   ⏱️ 실행시간: {execution_time:.4f}초")
            print(f"   🚫 에러: {str(e)}")
            
            raise  # 에러를 다시 발생시켜야 함
    
    return wrapper
```

#### 2️⃣ 고급 기능 추가
```python
def measure_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        # 메모리 사용량도 측정 (선택사항)
        import psutil
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            end_memory = process.memory_info().rss / 1024 / 1024
            
            print(f"✅ [{func.__name__}]")
            print(f"   ⏱️ 시간: {end_time - start_time:.4f}초")
            print(f"   💾 메모리: {end_memory - start_memory:.2f}MB")
            
            return result
            
        except Exception as e:
            print(f"❌ [{func.__name__}] 에러: {str(e)}")
            raise
    
    return wrapper
```

---

## 🔧 **문제 3: DataProcessor 클래스 힌트**

### 단계별 구현 가이드

#### 1️⃣ __init__ 메서드
```python
def __init__(self, input_file, output_file, timeout=30):
    self.input_file = input_file
    self.output_file = output_file
    self.timeout = timeout
    self.start_time = None
    self.log_file = f"{input_file}_processing.log"
```

#### 2️⃣ 컨텍스트 매니저 메서드
```python
def __enter__(self):
    # 처리 시작 시간 기록
    self.start_time = time.time()
    
    # 로그 파일 생성
    with open(self.log_file, 'w') as f:
        f.write(f"처리 시작: {time.ctime()}\n")
        f.write(f"입력 파일: {self.input_file}\n")
        f.write(f"출력 파일: {self.output_file}\n")
        f.write("-" * 50 + "\n")
    
    return self

def __exit__(self, exc_type, exc_val, exc_tb):
    end_time = time.time()
    processing_time = end_time - self.start_time
    
    with open(self.log_file, 'a') as f:
        f.write(f"\n처리 완료: {time.ctime()}\n")
        f.write(f"총 처리 시간: {processing_time:.2f}초\n")
        
        if exc_type:
            f.write(f"❌ 에러 발생: {exc_type.__name__}: {exc_val}\n")
        else:
            f.write("✅ 성공적으로 완료\n")
```

#### 3️⃣ process_line 메서드
```python
def process_line(self, line):
    # 빈 줄 체크
    if not line.strip():
        raise InvalidDataFormatError("빈 줄은 처리할 수 없습니다")
    
    # 기본적인 데이터 정제
    cleaned_line = line.strip()
    
    # 특정 형식 검증 (예: CSV 형태)
    if ',' not in cleaned_line:
        raise InvalidDataFormatError("CSV 형식이 아닙니다")
    
    # 처리된 데이터 반환
    return cleaned_line.upper()  # 예시: 대문자 변환
```

#### 4️⃣ process_file 메서드
```python
def process_file(self):
    try:
        with open(self.input_file, 'r') as infile, \
             open(self.output_file, 'w') as outfile:
            
            line_count = 0
            error_count = 0
            
            for line_num, line in enumerate(infile, 1):
                try:
                    # 타임아웃 체크
                    if time.time() - self.start_time > self.timeout:
                        raise ProcessingTimeoutError(f"처리 시간 초과: {self.timeout}초")
                    
                    # 줄 처리
                    processed_line = self.process_line(line)
                    outfile.write(processed_line + '\n')
                    line_count += 1
                    
                except InvalidDataFormatError as e:
                    error_count += 1
                    with open(self.log_file, 'a') as log:
                        log.write(f"라인 {line_num} 에러: {str(e)}\n")
            
            print(f"✅ 처리 완료: {line_count}줄 성공, {error_count}줄 에러")
            
    except FileNotFoundError:
        raise DataProcessingError(f"입력 파일을 찾을 수 없습니다: {self.input_file}")
```

---

## 🏆 **보너스 문제: CSVAnalyzer 힌트**

### 핵심 아이디어
1. **CSVReader를 상속**하거나 **조합**으로 사용
2. **measure_time 데코레이터** 적용
3. **컨텍스트 매니저** 구현
4. **통계 계산** 로직

```python
class CSVAnalyzer:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.reader = CSVReader(csv_file)  # 조합 방식
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        print(f"📊 CSV 분석 시작: {self.csv_file}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        print(f"📊 분석 완료: {end_time - self.start_time:.2f}초")
    
    @measure_time
    def analyze_numeric_column(self, column_name):
        # 1. 컬럼 인덱스 찾기
        headers = self.reader.headers
        if column_name not in headers:
            raise ValueError(f"컬럼 '{column_name}'을 찾을 수 없습니다")
        
        col_index = headers.index(column_name)
        
        # 2. 숫자 데이터 추출
        values = []
        for row in self.reader.data[1:]:  # 헤더 제외
            try:
                values.append(float(row[col_index]))
            except (ValueError, IndexError):
                continue  # 숫자가 아닌 값 무시
        
        # 3. 통계 계산
        if not values:
            return {"error": "숫자 데이터가 없습니다"}
        
        return {
            "컬럼명": column_name,
            "데이터_개수": len(values),
            "평균": sum(values) / len(values),
            "최대값": max(values),
            "최소값": min(values),
            "합계": sum(values)
        }
```

---

## 💡 **추가 팁**

### 1. 에러 처리 패턴
```python
try:
    # 위험한 작업
    result = risky_operation()
except SpecificError as e:
    # 특정 에러 처리
    log_error(e)
    raise
except Exception as e:
    # 일반 에러 처리
    log_error(e)
    raise CustomError(f"예상치 못한 에러: {e}")
finally:
    # 정리 작업
    cleanup()
```

### 2. 파일 처리 패턴
```python
# 안전한 파일 처리
with open(filename, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, 1):
        try:
            process_line(line)
        except Exception as e:
            print(f"라인 {line_num} 에러: {e}")
```

### 3. 성능 최적화
```python
# 대용량 데이터 처리
def process_large_file(filename, chunk_size=1024):
    with open(filename, 'r') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield process_chunk(chunk)
```

## 🎯 **완료 체크리스트**
- [ ] CSVReader 클래스 구현 완료
- [ ] measure_time 데코레이터 작동 확인  
- [ ] DataProcessor 컨텍스트 매니저 테스트
- [ ] 모든 예외 처리 케이스 확인
- [ ] 실제 CSV 파일로 테스트
- [ ] 보너스 문제 도전

완료하시면 2단계로 넘어갑니다! 💪