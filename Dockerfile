# Use a Python image with uv pre-installed
FROM astral/uv:python3.12-bookworm-slim

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

# Add virtual environment to PATH (MUST be before playwright install)
ENV PATH="/app/.venv/bin:$PATH"

# Install Playwright browsers as ROOT with verbose output
RUN echo "=== Installing Playwright browsers ===" && \
    uv run playwright install chromium --with-deps && \
    echo "=== Verifying installation ===" && \
    ls -la /root/.cache/ms-playwright/ && \
    if [ ! -d "/root/.cache/ms-playwright/chromium"* ]; then \
        echo "ERROR: Chromium not found!"; \
        exit 1; \
    fi && \
    echo "✅ Playwright browsers installed successfully"

# Create non-root user
RUN useradd -m -u 1000 quizuser && \
    mkdir -p /app/data/quiz-jobs && \
    chown -R quizuser:quizuser /app

# Set Playwright to use root's browser cache
ENV PLAYWRIGHT_BROWSERS_PATH=/root/.cache/ms-playwright

# Give quizuser read access to browser cache
RUN chmod -R 755 /root && \
    chmod -R 755 /root/.cache

# Switch to non-root user
USER quizuser

# Expose port
EXPOSE 8000

# Reset entrypoint
ENTRYPOINT []

# Run FastAPI application
CMD ["sh", "-c", "uv run uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

