FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PR_MODEL_DIR=/app/artifacts
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .
COPY configs ./configs
COPY artifacts ./artifacts
USER appuser
EXPOSE 8080
CMD ["uvicorn", "prembly_risk.api:app", "--host", "0.0.0.0", "--port", "8080"]

