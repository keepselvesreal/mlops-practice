# mlflow-server/Dockerfile
FROM python:3.10-slim

WORKDIR /mlflow

# 필요한 패키지 설치
RUN apt-get update && apt-get install -y curl && \
    pip install --no-cache-dir mlflow psycopg2-binary

# 환경 변수 설정
ENV BACKEND_STORE_URI=sqlite:///mlflow.db
# ENV ARTIFACT_ROOT=/mlflow/artifacts

# 실행 명령
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:///mlflow.db", "--default-artifact-root", "/mlflow/artifacts"]
# CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:///mlflow.db"]