FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY config ./config
COPY src ./src
COPY dashboard ./dashboard

EXPOSE 8000

CMD ["uv", "run", "--no-dev", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
