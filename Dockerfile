FROM python:3.13-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_DOWNLOADS=never
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

FROM python:3.13-slim
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV PROJECT_ROOT=/app
COPY src ./src
COPY scripts ./scripts
COPY data ./data
COPY artifacts ./artifacts
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser
CMD ["python", "scripts/score_holdout.py", "data/sample_holdout.csv", "/tmp/predictions.csv"]
