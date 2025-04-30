FROM python:3.10-slim

WORKDIR /app

# Install system dependencies and Docker CLI
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    apt-transport-https \
    ca-certificates \
    gnupg \
    && curl -fsSL https://get.docker.com -o get-docker.sh \
    && sh get-docker.sh \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better cache utilization
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the pipeline code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "agent.py"]