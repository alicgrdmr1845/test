FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY server/requirements.txt /app/server/
RUN pip install --no-cache-dir -r server/requirements.txt

# Copy application code
COPY server/ /app/server/

# Create necessary directories
RUN mkdir -p /app/storage /app/logs /app/intake

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "-m", "uvicorn", "server.api:app", "--host", "0.0.0.0", "--port", "8000"]