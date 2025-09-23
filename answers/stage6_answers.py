"""
6단계: 프로덕션 배포 답안 스크립트
"""

import os
import yaml
import json
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 문제 1: Airflow DAG 구성
def problem1_solution():
    """Airflow DAG 구성"""

    def create_airflow_dag():
        """데이터 파이프라인을 Airflow DAG로 구현"""

        dag_content = """
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.utils.dates import days_ago
from airflow.utils.email import send_email
import pandas as pd
import logging

# 기본 인수 설정
default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'email': ['data-team@company.com']
}

# DAG 정의
dag = DAG(
    'data_production_pipeline',
    default_args=default_args,
    description='Daily data production pipeline',
    schedule_interval='0 2 * * *',  # 매일 새벽 2시 실행
    catchup=False,
    max_active_runs=1,
    tags=['data-production', 'etl']
)

def extract_data(**context):
    \"\"\"데이터 추출 태스크\"\"\"
    execution_date = context['execution_date']
    logger = logging.getLogger(__name__)

    try:
        logger.info(f"Starting data extraction for {execution_date}")

        # 실제 데이터 추출 로직
        # 예: 데이터베이스에서 어제 데이터 추출
        query = f\"\"\"
            SELECT * FROM sales_data
            WHERE date = '{execution_date.date()}'
        \"\"\"

        # 시뮬레이션: 추출된 데이터 크기
        extracted_rows = 150000

        logger.info(f"Data extraction completed: {extracted_rows} rows")

        # XCom으로 다음 태스크에 전달
        return {'extracted_rows': extracted_rows, 'status': 'success'}

    except Exception as e:
        logger.error(f"Data extraction failed: {str(e)}")
        send_email(
            to=['data-team@company.com'],
            subject='Data Extraction Failed',
            html_content=f'Data extraction failed with error: {str(e)}'
        )
        raise

def validate_data(**context):
    \"\"\"데이터 검증 태스크\"\"\"
    logger = logging.getLogger(__name__)

    # 이전 태스크 결과 가져오기
    ti = context['task_instance']
    extract_result = ti.xcom_pull(task_ids='extract_data')

    try:
        logger.info("Starting data validation")

        # 데이터 품질 검사
        validation_rules = [
            'no_null_customer_id',
            'positive_amounts',
            'valid_date_range'
        ]

        validation_results = {}
        for rule in validation_rules:
            # 시뮬레이션: 검증 결과
            validation_results[rule] = True

        failed_validations = [rule for rule, result in validation_results.items() if not result]

        if failed_validations:
            raise ValueError(f"Validation failed for rules: {failed_validations}")

        logger.info("Data validation completed successfully")
        return {'validation_status': 'passed', 'validated_rows': extract_result['extracted_rows']}

    except Exception as e:
        logger.error(f"Data validation failed: {str(e)}")
        raise

def transform_data(**context):
    \"\"\"데이터 변환 태스크\"\"\"
    logger = logging.getLogger(__name__)

    ti = context['task_instance']
    validation_result = ti.xcom_pull(task_ids='validate_data')

    try:
        logger.info("Starting data transformation")

        # 변환 로직 실행
        transformations = [
            'standardize_currency',
            'calculate_metrics',
            'enrich_customer_data'
        ]

        transformed_rows = validation_result['validated_rows']

        for transformation in transformations:
            logger.info(f"Applying transformation: {transformation}")
            # 실제 변환 로직

        logger.info(f"Data transformation completed: {transformed_rows} rows")
        return {'transformed_rows': transformed_rows, 'status': 'success'}

    except Exception as e:
        logger.error(f"Data transformation failed: {str(e)}")
        raise

def load_data(**context):
    \"\"\"데이터 적재 태스크\"\"\"
    logger = logging.getLogger(__name__)

    ti = context['task_instance']
    transform_result = ti.xcom_pull(task_ids='transform_data')

    try:
        logger.info("Starting data loading")

        # 데이터 웨어하우스에 적재
        loaded_rows = transform_result['transformed_rows']

        logger.info(f"Data loading completed: {loaded_rows} rows")
        return {'loaded_rows': loaded_rows, 'status': 'success'}

    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")
        raise

def send_success_notification(**context):
    \"\"\"성공 알림 전송\"\"\"
    ti = context['task_instance']
    load_result = ti.xcom_pull(task_ids='load_data')

    send_email(
        to=['data-team@company.com'],
        subject='Data Pipeline Completed Successfully',
        html_content=f\"\"\"
        <h3>Data Pipeline Execution Summary</h3>
        <p>Execution Date: {context['execution_date']}</p>
        <p>Loaded Rows: {load_result['loaded_rows']}</p>
        <p>Status: Success</p>
        \"\"\"
    )

# 태스크 정의
start_task = DummyOperator(task_id='start', dag=dag)

extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    dag=dag
)

validate_task = PythonOperator(
    task_id='validate_data',
    python_callable=validate_data,
    dag=dag
)

transform_task = PythonOperator(
    task_id='transform_data',
    python_callable=transform_data,
    dag=dag
)

load_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    dag=dag
)

# 데이터 품질 체크 (병렬 실행)
quality_check_task = BashOperator(
    task_id='quality_check',
    bash_command='python /opt/airflow/scripts/quality_check.py',
    dag=dag
)

# 백업 태스크 (병렬 실행)
backup_task = BashOperator(
    task_id='backup_data',
    bash_command='python /opt/airflow/scripts/backup.py',
    dag=dag
)

notification_task = PythonOperator(
    task_id='send_notification',
    python_callable=send_success_notification,
    dag=dag,
    trigger_rule='all_success'
)

end_task = DummyOperator(task_id='end', dag=dag)

# 태스크 의존성 정의
start_task >> extract_task >> validate_task >> transform_task >> load_task

# 병렬 태스크
load_task >> [quality_check_task, backup_task] >> notification_task >> end_task

# 실패 시 알림 태스크
failure_notification_task = PythonOperator(
    task_id='failure_notification',
    python_callable=lambda **context: send_email(
        to=['data-team@company.com'],
        subject='Data Pipeline Failed',
        html_content=f'Pipeline failed at {context["execution_date"]}'
    ),
    dag=dag,
    trigger_rule='one_failed'
)

# 모든 태스크가 실패하면 알림
[extract_task, validate_task, transform_task, load_task] >> failure_notification_task
"""

        # DAG 파일 저장
        with open('data_production_pipeline.py', 'w', encoding='utf-8') as f:
            f.write(dag_content)

        logger.info("Airflow DAG 파일 생성 완료: data_production_pipeline.py")

        return {
            'dag_id': 'data_production_pipeline',
            'schedule': '0 2 * * *',
            'tasks': ['extract_data', 'validate_data', 'transform_data', 'load_data'],
            'features': ['retry_logic', 'email_notifications', 'parallel_execution']
        }

    return create_airflow_dag()

# 문제 2: Docker 컨테이너화
def problem2_solution():
    """Docker 컨테이너화"""

    def create_docker_files():
        """Docker 관련 파일 생성"""

        # Dockerfile (멀티스테이지 빌드)
        dockerfile_content = """
# 빌드 스테이지
FROM python:3.9-slim as builder

WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# 프로덕션 스테이지
FROM python:3.9-slim

# 보안을 위한 non-root 사용자 생성
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# 시스템 의존성 (런타임에 필요한 것만)
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# 빌드 스테이지에서 Python 패키지 복사
COPY --from=builder /root/.local /home/appuser/.local

# 애플리케이션 코드 복사
COPY --chown=appuser:appuser . .

# PATH 설정
ENV PATH=/home/appuser/.local/bin:$PATH

# 환경 변수
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 포트 노출
EXPOSE 8000

# 헬스체크
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# 사용자 변경
USER appuser

# 애플리케이션 시작
CMD ["python", "app.py"]
"""

        # requirements.txt
        requirements_content = """
pandas==1.5.3
numpy==1.24.3
sqlalchemy==2.0.15
psycopg2-binary==2.9.6
redis==4.5.5
celery==5.2.7
flask==2.3.2
gunicorn==20.1.0
prometheus-client==0.16.0
structlog==23.1.0
"""

        # docker-compose.yml
        docker_compose_content = """
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: data-processor
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
      - LOG_LEVEL=${LOG_LEVEL:-INFO}
      - WORKER_PROCESSES=${WORKER_PROCESSES:-2}
    volumes:
      - ./data:/app/data:ro
      - ./logs:/app/logs
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    networks:
      - data-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  postgres:
    image: postgres:15-alpine
    container_name: postgres-db
    environment:
      - POSTGRES_DB=${POSTGRES_DB}
      - POSTGRES_USER=${POSTGRES_USER}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    restart: unless-stopped
    networks:
      - data-network

  redis:
    image: redis:7-alpine
    container_name: redis-cache
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    networks:
      - data-network
    command: redis-server --appendonly yes

  worker:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: celery-worker
    command: celery -A app.celery worker --loglevel=info
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
    volumes:
      - ./data:/app/data:ro
      - ./logs:/app/logs
    depends_on:
      - redis
      - postgres
    restart: unless-stopped
    networks:
      - data-network

  scheduler:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: celery-beat
    command: celery -A app.celery beat --loglevel=info
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
    volumes:
      - ./logs:/app/logs
    depends_on:
      - redis
      - postgres
    restart: unless-stopped
    networks:
      - data-network

  monitoring:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'
    restart: unless-stopped
    networks:
      - data-network

volumes:
  postgres_data:
  redis_data:
  prometheus_data:

networks:
  data-network:
    driver: bridge
"""

        # .env 파일 템플릿
        env_content = """
# 데이터베이스 설정
DATABASE_URL=postgresql://datauser:datapass@postgres:5432/datadb
POSTGRES_DB=datadb
POSTGRES_USER=datauser
POSTGRES_PASSWORD=datapass

# Redis 설정
REDIS_URL=redis://redis:6379/0

# 애플리케이션 설정
LOG_LEVEL=INFO
WORKER_PROCESSES=2
DEBUG=False

# 보안 설정
SECRET_KEY=your-secret-key-here
JWT_SECRET=your-jwt-secret-here
"""

        # .dockerignore
        dockerignore_content = """
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/
pip-log.txt
pip-delete-this-directory.txt
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.git/
.mypy_cache/
.pytest_cache/
.hypothesis/
.DS_Store
*.egg-info/
dist/
build/
"""

        # 파일들 저장
        files = {
            'Dockerfile': dockerfile_content,
            'requirements.txt': requirements_content,
            'docker-compose.yml': docker_compose_content,
            '.env.template': env_content,
            '.dockerignore': dockerignore_content
        }

        for filename, content in files.items():
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)

        logger.info(f"Docker 설정 파일 {len(files)}개 생성 완료")

        return {
            'files_created': list(files.keys()),
            'features': [
                'multi_stage_build',
                'non_root_user',
                'health_checks',
                'environment_variables',
                'volume_mounts',
                'monitoring_integration'
            ]
        }

    return create_docker_files()

# 문제 3: CI/CD 파이프라인
def problem3_solution():
    """CI/CD 파이프라인"""

    def create_github_actions_workflow():
        """GitHub Actions 워크플로우 생성"""

        workflow_content = """
name: Data Production Pipeline CI/CD

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 6 * * 1'  # 매주 월요일 6시

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest

    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_db
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432

      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379

    steps:
    - name: Checkout code
      uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
        cache: 'pip'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt

    - name: Lint with flake8
      run: |
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

    - name: Type check with mypy
      run: |
        mypy src/ --ignore-missing-imports

    - name: Security scan with bandit
      run: |
        bandit -r src/ -f json -o bandit-report.json
      continue-on-error: true

    - name: Run tests with pytest
      run: |
        pytest tests/ --cov=src --cov-report=xml --cov-report=html --junitxml=test-results.xml
      env:
        DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_db
        REDIS_URL: redis://localhost:6379/0

    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella

    - name: Upload test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: test-results
        path: |
          test-results.xml
          htmlcov/
          bandit-report.json

  security:
    runs-on: ubuntu-latest
    steps:
    - name: Checkout code
      uses: actions/checkout@v3

    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        scan-ref: '.'
        format: 'sarif'
        output: 'trivy-results.sarif'

    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'

  build:
    needs: [test, security]
    runs-on: ubuntu-latest

    outputs:
      image: ${{ steps.image.outputs.image }}
      digest: ${{ steps.build.outputs.digest }}

    steps:
    - name: Checkout code
      uses: actions/checkout@v3

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2

    - name: Log in to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}

    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=sha,prefix={{branch}}-
          type=raw,value=latest,enable={{is_default_branch}}

    - name: Build and push Docker image
      id: build
      uses: docker/build-push-action@v4
      with:
        context: .
        platforms: linux/amd64,linux/arm64
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

    - name: Output image
      id: image
      run: |
        echo "image=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}" >> $GITHUB_OUTPUT

  deploy-staging:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/develop'
    environment: staging

    steps:
    - name: Checkout code
      uses: actions/checkout@v3

    - name: Deploy to staging
      run: |
        echo "Deploying to staging environment"
        # 실제 배포 명령어
        # kubectl set image deployment/data-processor data-processor=${{ needs.build.outputs.image }}@${{ needs.build.outputs.digest }}

    - name: Run integration tests
      run: |
        echo "Running integration tests on staging"
        # 실제 통합 테스트 실행

    - name: Performance tests
      run: |
        echo "Running performance tests"
        # 성능 테스트 실행

  deploy-production:
    needs: [build, deploy-staging]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: production

    steps:
    - name: Checkout code
      uses: actions/checkout@v3

    - name: Deploy to production
      run: |
        echo "Deploying to production environment"
        # Blue-Green 배포 또는 Rolling 업데이트

    - name: Health check
      run: |
        echo "Performing health check"
        # 프로덕션 헬스 체크

    - name: Notify deployment
      uses: 8398a7/action-slack@v3
      with:
        status: ${{ job.status }}
        channel: '#deployments'
        webhook_url: ${{ secrets.SLACK_WEBHOOK }}

  rollback:
    runs-on: ubuntu-latest
    if: failure() && github.ref == 'refs/heads/main'
    needs: deploy-production
    environment: production

    steps:
    - name: Rollback deployment
      run: |
        echo "Rolling back to previous version"
        # 롤백 로직

    - name: Notify rollback
      uses: 8398a7/action-slack@v3
      with:
        status: 'warning'
        channel: '#deployments'
        webhook_url: ${{ secrets.SLACK_WEBHOOK }}
        message: 'Production deployment rolled back due to failure'
"""

        # requirements-dev.txt
        dev_requirements_content = """
pytest==7.3.1
pytest-cov==4.1.0
flake8==6.0.0
mypy==1.3.0
bandit==1.7.5
black==23.3.0
isort==5.12.0
pre-commit==3.3.2
"""

        # pytest.ini
        pytest_config_content = """
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_functions = test_*
python_classes = Test*
addopts =
    --strict-markers
    --disable-warnings
    --tb=short
    -v
markers =
    unit: Unit tests
    integration: Integration tests
    slow: Slow running tests
"""

        # .pre-commit-config.yaml
        precommit_config_content = """
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3.9

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
"""

        # 파일들 저장
        os.makedirs('.github/workflows', exist_ok=True)

        files = {
            '.github/workflows/ci-cd.yml': workflow_content,
            'requirements-dev.txt': dev_requirements_content,
            'pytest.ini': pytest_config_content,
            '.pre-commit-config.yaml': precommit_config_content
        }

        for filepath, content in files.items():
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

        logger.info("CI/CD 파이프라인 설정 파일 생성 완료")

        return {
            'pipeline_stages': ['test', 'security', 'build', 'deploy-staging', 'deploy-production'],
            'quality_gates': ['linting', 'type_checking', 'security_scan', 'unit_tests', 'integration_tests'],
            'deployment_strategy': 'blue_green_with_rollback',
            'environments': ['staging', 'production']
        }

    return create_github_actions_workflow()

# 문제 4: 환경 관리
def problem4_solution():
    """환경 관리"""

    def create_environment_configs():
        """환경별 설정 파일 생성"""

        # 개발 환경 설정
        dev_config = {
            'database': {
                'host': 'localhost',
                'port': 5432,
                'name': 'dev_datadb',
                'user': 'dev_user',
                'pool_size': 5
            },
            'redis': {
                'host': 'localhost',
                'port': 6379,
                'db': 0
            },
            'logging': {
                'level': 'DEBUG',
                'format': 'detailed',
                'file_rotation': False
            },
            'features': {
                'debug_mode': True,
                'hot_reload': True,
                'profiling': True
            },
            'batch_size': 1000,
            'worker_processes': 1
        }

        # 스테이징 환경 설정
        staging_config = {
            'database': {
                'host': 'staging-db.company.com',
                'port': 5432,
                'name': 'staging_datadb',
                'user': 'staging_user',
                'pool_size': 10
            },
            'redis': {
                'host': 'staging-redis.company.com',
                'port': 6379,
                'db': 0
            },
            'logging': {
                'level': 'INFO',
                'format': 'json',
                'file_rotation': True
            },
            'features': {
                'debug_mode': False,
                'hot_reload': False,
                'profiling': False
            },
            'batch_size': 5000,
            'worker_processes': 2
        }

        # 프로덕션 환경 설정
        prod_config = {
            'database': {
                'host': 'prod-db.company.com',
                'port': 5432,
                'name': 'prod_datadb',
                'user': 'prod_user',
                'pool_size': 20,
                'ssl_mode': 'require'
            },
            'redis': {
                'host': 'prod-redis.company.com',
                'port': 6379,
                'db': 0,
                'ssl': True
            },
            'logging': {
                'level': 'WARNING',
                'format': 'json',
                'file_rotation': True,
                'retention_days': 30
            },
            'features': {
                'debug_mode': False,
                'hot_reload': False,
                'profiling': False
            },
            'batch_size': 10000,
            'worker_processes': 4,
            'monitoring': {
                'enabled': True,
                'metrics_endpoint': '/metrics',
                'health_endpoint': '/health'
            }
        }

        # 환경별 Docker Compose 오버라이드
        dev_compose_override = """
version: '3.8'

services:
  app:
    build:
      context: .
      target: builder  # 개발용 빌드 타겟
    environment:
      - DEBUG=True
      - LOG_LEVEL=DEBUG
    volumes:
      - ./src:/app/src:ro  # 코드 핫 리로드를 위한 볼륨
      - ./tests:/app/tests:ro
    ports:
      - "8000:8000"
      - "5678:5678"  # 디버거 포트

  postgres:
    environment:
      - POSTGRES_DB=dev_datadb
    ports:
      - "5432:5432"  # 로컬 접근을 위해 포트 노출

  redis:
    ports:
      - "6379:6379"  # 로컬 접근을 위해 포트 노출
"""

        staging_compose_override = """
version: '3.8'

services:
  app:
    environment:
      - DEBUG=False
      - LOG_LEVEL=INFO
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 512M
        reservations:
          memory: 256M

  postgres:
    environment:
      - POSTGRES_DB=staging_datadb
    deploy:
      resources:
        limits:
          memory: 1G
"""

        prod_compose_override = """
version: '3.8'

services:
  app:
    environment:
      - DEBUG=False
      - LOG_LEVEL=WARNING
    deploy:
      replicas: 4
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
        reservations:
          memory: 512M
          cpus: '0.25'
    restart: always

  postgres:
    environment:
      - POSTGRES_DB=prod_datadb
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
"""

        # 설정 관리 클래스
        config_manager_content = """
import os
import yaml
import json
from typing import Dict, Any
from pathlib import Path

class ConfigManager:
    def __init__(self, environment: str = None):
        self.environment = environment or os.getenv('ENVIRONMENT', 'development')
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        \"\"\"환경에 따른 설정 로드\"\"\"
        config_file = f'config/{self.environment}.yml'

        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        # 환경 변수로 오버라이드
        config = self._override_with_env_vars(config)

        return config

    def _override_with_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        \"\"\"환경 변수로 설정 오버라이드\"\"\"
        env_mappings = {
            'DATABASE_HOST': 'database.host',
            'DATABASE_PORT': 'database.port',
            'DATABASE_NAME': 'database.name',
            'DATABASE_USER': 'database.user',
            'REDIS_HOST': 'redis.host',
            'REDIS_PORT': 'redis.port',
            'LOG_LEVEL': 'logging.level',
            'BATCH_SIZE': 'batch_size',
            'WORKER_PROCESSES': 'worker_processes'
        }

        for env_var, config_path in env_mappings.items():
            if env_var in os.environ:
                self._set_nested_value(config, config_path, os.environ[env_var])

        return config

    def _set_nested_value(self, config: Dict, path: str, value: str):
        \"\"\"중첩된 딕셔너리에 값 설정\"\"\"
        keys = path.split('.')
        current = config

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # 타입 변환
        if value.isdigit():
            value = int(value)
        elif value.lower() in ['true', 'false']:
            value = value.lower() == 'true'

        current[keys[-1]] = value

    def get(self, key: str, default: Any = None) -> Any:
        \"\"\"설정 값 가져오기\"\"\"
        keys = key.split('.')
        current = self.config

        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default

        return current

    def get_database_url(self) -> str:
        \"\"\"데이터베이스 URL 생성\"\"\"
        db_config = self.config['database']
        password = os.getenv('DATABASE_PASSWORD', '')

        return (f"postgresql://{db_config['user']}:{password}@"
                f"{db_config['host']}:{db_config['port']}/{db_config['name']}")

    def get_redis_url(self) -> str:
        \"\"\"Redis URL 생성\"\"\"
        redis_config = self.config['redis']
        return f"redis://{redis_config['host']}:{redis_config['port']}/{redis_config['db']}"

# 사용 예시
if __name__ == "__main__":
    config = ConfigManager()
    print(f"Environment: {config.environment}")
    print(f"Database URL: {config.get_database_url()}")
    print(f"Batch Size: {config.get('batch_size')}")
"""

        # 시크릿 관리 예시
        secrets_manager_content = """
import os
import json
from typing import Optional, Dict, Any

class SecretsManager:
    \"\"\"시크릿 관리 클래스 (실제로는 AWS Secrets Manager, HashiCorp Vault 등 사용)\"\"\"

    def __init__(self):
        self.secrets_cache = {}

    def get_secret(self, secret_name: str) -> Optional[str]:
        \"\"\"시크릿 값 가져오기\"\"\"
        # 캐시에서 먼저 확인
        if secret_name in self.secrets_cache:
            return self.secrets_cache[secret_name]

        # 환경 변수에서 확인
        env_value = os.getenv(secret_name)
        if env_value:
            self.secrets_cache[secret_name] = env_value
            return env_value

        # 파일에서 확인 (로컬 개발용)
        secret_file = f'/run/secrets/{secret_name}'
        if os.path.exists(secret_file):
            with open(secret_file, 'r') as f:
                value = f.read().strip()
                self.secrets_cache[secret_name] = value
                return value

        return None

    def get_database_password(self) -> str:
        \"\"\"데이터베이스 패스워드 가져오기\"\"\"
        return self.get_secret('DATABASE_PASSWORD') or 'default_password'

    def get_api_key(self, service: str) -> str:
        \"\"\"API 키 가져오기\"\"\"
        return self.get_secret(f'{service.upper()}_API_KEY') or ''

# Docker Secrets를 위한 Compose 설정
secrets_compose_content = '''
version: '3.8'

services:
  app:
    secrets:
      - database_password
      - api_key
    environment:
      - DATABASE_PASSWORD_FILE=/run/secrets/database_password

secrets:
  database_password:
    file: ./secrets/database_password.txt
  api_key:
    file: ./secrets/api_key.txt
'''
"""

        # 환경별 테스트 설정
        test_config_content = """
import pytest
import os
from src.config import ConfigManager

class TestEnvironmentConfigs:

    def test_development_config(self):
        os.environ['ENVIRONMENT'] = 'development'
        config = ConfigManager()

        assert config.get('logging.level') == 'DEBUG'
        assert config.get('features.debug_mode') is True
        assert config.get('batch_size') == 1000

    def test_staging_config(self):
        os.environ['ENVIRONMENT'] = 'staging'
        config = ConfigManager()

        assert config.get('logging.level') == 'INFO'
        assert config.get('features.debug_mode') is False
        assert config.get('batch_size') == 5000

    def test_production_config(self):
        os.environ['ENVIRONMENT'] = 'production'
        config = ConfigManager()

        assert config.get('logging.level') == 'WARNING'
        assert config.get('features.debug_mode') is False
        assert config.get('batch_size') == 10000

    def test_environment_variable_override(self):
        os.environ['ENVIRONMENT'] = 'development'
        os.environ['DATABASE_HOST'] = 'override_host'
        os.environ['BATCH_SIZE'] = '9999'

        config = ConfigManager()

        assert config.get('database.host') == 'override_host'
        assert config.get('batch_size') == 9999
"""

        # 파일들 저장
        os.makedirs('config', exist_ok=True)
        os.makedirs('src', exist_ok=True)
        os.makedirs('tests', exist_ok=True)

        files = {
            'config/development.yml': yaml.dump(dev_config, default_flow_style=False),
            'config/staging.yml': yaml.dump(staging_config, default_flow_style=False),
            'config/production.yml': yaml.dump(prod_config, default_flow_style=False),
            'docker-compose.dev.yml': dev_compose_override,
            'docker-compose.staging.yml': staging_compose_override,
            'docker-compose.prod.yml': prod_compose_override,
            'src/config.py': config_manager_content,
            'src/secrets.py': secrets_manager_content,
            'tests/test_config.py': test_config_content
        }

        for filepath, content in files.items():
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

        logger.info("환경 관리 설정 파일 생성 완료")

        return {
            'environments': ['development', 'staging', 'production'],
            'config_features': ['environment_separation', 'secret_management', 'override_support'],
            'files_created': len(files)
        }

    return create_environment_configs()

# 문제 5: 모니터링 및 알림
def problem5_solution():
    """모니터링 및 알림"""

    def create_monitoring_setup():
        """모니터링 및 알림 시스템 설정"""

        # Prometheus 설정
        prometheus_config = """
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'data-processor'
    static_configs:
      - targets: ['app:8000']
    metrics_path: '/metrics'
    scrape_interval: 15s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']
"""

        # 알림 규칙
        alert_rules_content = """
groups:
  - name: data_processing_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors per second"

      - alert: DatabaseConnectionFailed
        expr: up{job="postgres"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Database connection failed"
          description: "Cannot connect to PostgreSQL database"

      - alert: HighMemoryUsage
        expr: (process_resident_memory_bytes / 1024 / 1024) > 500
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value }}MB"

      - alert: DataProcessingDelay
        expr: data_processing_duration_seconds > 300
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Data processing taking too long"
          description: "Processing duration is {{ $value }} seconds"

      - alert: DataQualityIssue
        expr: data_quality_score < 80
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Data quality below threshold"
          description: "Data quality score is {{ $value }}%"
"""

        # Alertmanager 설정
        alertmanager_config = """
global:
  smtp_smarthost: 'smtp.company.com:587'
  smtp_from: 'alerts@company.com'
  smtp_auth_username: 'alerts@company.com'
  smtp_auth_password: 'smtp_password'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'
  routes:
    - match:
        severity: critical
      receiver: 'critical-alerts'
    - match:
        severity: warning
      receiver: 'warning-alerts'

receivers:
  - name: 'web.hook'
    webhook_configs:
      - url: 'http://localhost:5001/'

  - name: 'critical-alerts'
    email_configs:
      - to: 'data-team@company.com'
        subject: 'CRITICAL: {{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          {{ end }}
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
        channel: '#alerts-critical'
        title: 'Critical Alert'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'

  - name: 'warning-alerts'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
        channel: '#alerts-warning'
        title: 'Warning Alert'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
"""

        # 애플리케이션 메트릭 수집 코드
        metrics_collector_content = """
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import functools
from typing import Callable, Any

# 메트릭 정의
request_count = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
request_duration = Histogram('http_request_duration_seconds', 'HTTP request duration')
data_processing_duration = Histogram('data_processing_duration_seconds', 'Data processing duration')
data_quality_score = Gauge('data_quality_score', 'Current data quality score')
active_connections = Gauge('database_connections_active', 'Active database connections')
memory_usage = Gauge('process_memory_usage_bytes', 'Process memory usage in bytes')

class MetricsCollector:
    def __init__(self):
        self.start_time = time.time()

    def record_request(self, method: str, endpoint: str, status: int):
        \"\"\"HTTP 요청 메트릭 기록\"\"\"
        request_count.labels(method=method, endpoint=endpoint, status=status).inc()

    def record_processing_time(self, duration: float):
        \"\"\"처리 시간 메트릭 기록\"\"\"
        data_processing_duration.observe(duration)

    def update_quality_score(self, score: float):
        \"\"\"데이터 품질 점수 업데이트\"\"\"
        data_quality_score.set(score)

    def update_memory_usage(self, usage_bytes: int):
        \"\"\"메모리 사용량 업데이트\"\"\"
        memory_usage.set(usage_bytes)

def monitor_execution_time(metric_name: str = None):
    \"\"\"실행 시간 모니터링 데코레이터\"\"\"
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                if metric_name:
                    globals()[metric_name].observe(duration)
                else:
                    data_processing_duration.observe(duration)
        return wrapper
    return decorator

# 사용 예시
@monitor_execution_time('data_processing_duration')
def process_data_batch(data):
    # 데이터 처리 로직
    time.sleep(2)  # 시뮬레이션
    return "processed"

# 메트릭 서버 시작
def start_metrics_server(port: int = 8001):
    start_http_server(port)
    print(f"Metrics server started on port {port}")
"""

        # Grafana 대시보드 설정
        grafana_dashboard = {
            "dashboard": {
                "id": None,
                "title": "Data Production Pipeline",
                "tags": ["data", "production"],
                "timezone": "browser",
                "panels": [
                    {
                        "id": 1,
                        "title": "Request Rate",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(http_requests_total[5m])",
                                "legendFormat": "{{method}} {{endpoint}}"
                            }
                        ]
                    },
                    {
                        "id": 2,
                        "title": "Error Rate",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(http_requests_total{status=~\"5..\"}[5m])",
                                "legendFormat": "5xx errors"
                            }
                        ]
                    },
                    {
                        "id": 3,
                        "title": "Data Processing Duration",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "histogram_quantile(0.95, rate(data_processing_duration_seconds_bucket[5m]))",
                                "legendFormat": "95th percentile"
                            }
                        ]
                    },
                    {
                        "id": 4,
                        "title": "Data Quality Score",
                        "type": "singlestat",
                        "targets": [
                            {
                                "expr": "data_quality_score",
                                "legendFormat": "Quality Score"
                            }
                        ]
                    }
                ],
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "refresh": "10s"
            }
        }

        # 로그 수집 설정 (Fluentd)
        fluentd_config = """
<source>
  @type tail
  path /app/logs/*.log
  pos_file /var/log/fluentd/app.log.pos
  tag app.logs
  format json
  time_key timestamp
  time_format %Y-%m-%dT%H:%M:%S.%L%z
</source>

<filter app.logs>
  @type parser
  key_name message
  <parse>
    @type json
  </parse>
</filter>

<match app.logs>
  @type elasticsearch
  host elasticsearch
  port 9200
  index_name app-logs
  type_name _doc
  <buffer>
    flush_interval 10s
  </buffer>
</match>
"""

        # 헬스체크 엔드포인트
        health_check_content = """
from flask import Flask, jsonify
import psutil
import redis
import psycopg2
from datetime import datetime

app = Flask(__name__)

class HealthChecker:
    def __init__(self):
        self.checks = {
            'database': self.check_database,
            'redis': self.check_redis,
            'memory': self.check_memory,
            'disk': self.check_disk_space
        }

    def check_database(self):
        \"\"\"데이터베이스 연결 확인\"\"\"
        try:
            conn = psycopg2.connect(
                host='postgres',
                database='datadb',
                user='datauser',
                password='datapass',
                connect_timeout=5
            )
            conn.close()
            return {'status': 'healthy', 'message': 'Database connection OK'}
        except Exception as e:
            return {'status': 'unhealthy', 'message': f'Database error: {str(e)}'}

    def check_redis(self):
        \"\"\"Redis 연결 확인\"\"\"
        try:
            r = redis.Redis(host='redis', port=6379, socket_timeout=5)
            r.ping()
            return {'status': 'healthy', 'message': 'Redis connection OK'}
        except Exception as e:
            return {'status': 'unhealthy', 'message': f'Redis error: {str(e)}'}

    def check_memory(self):
        \"\"\"메모리 사용량 확인\"\"\"
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            return {'status': 'unhealthy', 'message': f'High memory usage: {memory.percent}%'}
        return {'status': 'healthy', 'message': f'Memory usage: {memory.percent}%'}

    def check_disk_space(self):
        \"\"\"디스크 공간 확인\"\"\"
        disk = psutil.disk_usage('/')
        if disk.percent > 85:
            return {'status': 'unhealthy', 'message': f'Low disk space: {disk.percent}%'}
        return {'status': 'healthy', 'message': f'Disk usage: {disk.percent}%'}

    def run_all_checks(self):
        \"\"\"모든 헬스체크 실행\"\"\"
        results = {}
        overall_status = 'healthy'

        for check_name, check_func in self.checks.items():
            result = check_func()
            results[check_name] = result

            if result['status'] != 'healthy':
                overall_status = 'unhealthy'

        return {
            'status': overall_status,
            'timestamp': datetime.utcnow().isoformat(),
            'checks': results
        }

health_checker = HealthChecker()

@app.route('/health')
def health():
    result = health_checker.run_all_checks()
    status_code = 200 if result['status'] == 'healthy' else 503
    return jsonify(result), status_code

@app.route('/metrics')
def metrics():
    # Prometheus 메트릭 엔드포인트는 별도 포트에서 제공
    return "Metrics available on port 8001", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
"""

        # 파일들 저장
        files = {
            'prometheus.yml': prometheus_config,
            'alert_rules.yml': alert_rules_content,
            'alertmanager.yml': alertmanager_config,
            'src/metrics.py': metrics_collector_content,
            'grafana_dashboard.json': json.dumps(grafana_dashboard, indent=2),
            'fluentd.conf': fluentd_config,
            'src/health.py': health_check_content
        }

        for filepath, content in files.items():
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

        logger.info("모니터링 및 알림 시스템 설정 완료")

        return {
            'monitoring_components': ['prometheus', 'grafana', 'alertmanager', 'fluentd'],
            'metrics_types': ['request_metrics', 'processing_metrics', 'quality_metrics', 'system_metrics'],
            'alert_channels': ['email', 'slack', 'webhook'],
            'health_checks': ['database', 'redis', 'memory', 'disk']
        }

    return create_monitoring_setup()

if __name__ == "__main__":
    print("=== 문제 1: Airflow DAG 구성 ===")
    dag_result = problem1_solution()
    print(f"DAG 생성 완료: {dag_result['dag_id']}")

    print("\n=== 문제 2: Docker 컨테이너화 ===")
    docker_result = problem2_solution()
    print(f"Docker 파일 {len(docker_result['files_created'])}개 생성 완료")

    print("\n=== 문제 3: CI/CD 파이프라인 ===")
    cicd_result = problem3_solution()
    print(f"CI/CD 파이프라인 설정 완료: {len(cicd_result['pipeline_stages'])}단계")

    print("\n=== 문제 4: 환경 관리 ===")
    env_result = problem4_solution()
    print(f"환경 설정 완료: {len(env_result['environments'])}개 환경")

    print("\n=== 문제 5: 모니터링 및 알림 ===")
    monitoring_result = problem5_solution()
    print(f"모니터링 시스템 설정 완료: {len(monitoring_result['monitoring_components'])}개 컴포넌트")