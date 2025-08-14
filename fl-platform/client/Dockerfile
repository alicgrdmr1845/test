FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY client/requirements.txt /app/client/
RUN pip install --no-cache-dir -r client/requirements.txt

# Copy client code
COPY client/ /app/client/

# Default command
CMD ["python", "-m", "client.fl_client"]