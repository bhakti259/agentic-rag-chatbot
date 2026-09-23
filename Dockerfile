FROM python:3.12-slim

WORKDIR /app

# Install uv for dependency management
RUN pip install --no-cache-dir uv

# Copy dependency files first (better layer caching — rebuilds only
# reinstall dependencies if pyproject.toml/uv.lock actually changed)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy the rest of the application
COPY . .

# Streamlit's default port
EXPOSE 8501

# Healthcheck so orchestrators (Docker Compose, ECS, etc.) know the app is alive
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["uv", "run", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]