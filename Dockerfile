# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Set working directory
WORKDIR /app

# Install system dependencies for Playwright and PDF processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Playwright dependencies (minimal set)
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libdbus-1-3 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpango-1.0-0 \
    libcairo2 \
    # PDF processing
    poppler-utils \
    # Cleanup to reduce image size
    && rm -rf /var/lib/apt/lists/*

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking
ENV UV_LINK_MODE=copy

# Install only dependencies first (for Docker layer caching)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# Copy project files
COPY . /app

# Install project into the container
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Install Playwright browsers (Chromium only, headless)
# Note: This must run AFTER uv sync (playwright needs to be installed first)
RUN uv run playwright install chromium --with-deps

# Add virtual environment to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Create non-root user for security
RUN useradd -m -u 1000 quizuser && \
    chown -R quizuser:quizuser /app && \
    mkdir -p /tmp/quiz-jobs && \
    chown -R quizuser:quizuser /tmp/quiz-jobs

# Switch to non-root user
USER quizuser

# Expose port (for documentation, Azure Container Apps ignores this)
EXPOSE 8000

# Reset entrypoint (inherited from base image)
ENTRYPOINT []

# Run FastAPI application
CMD ["sh", "-c", "uv run uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

