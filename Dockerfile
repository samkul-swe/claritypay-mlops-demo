# Dockerfile

FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY models/ models/
COPY src/ src/
COPY data/ data/

# Expose port
EXPOSE 8000

# Run API
CMD ["python", "src/api/main.py"]