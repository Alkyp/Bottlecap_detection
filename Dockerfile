# ===== Base Image =====
FROM python:3.10-slim

# ===== Env Setup =====
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install basic dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget unzip build-essential && \
    rm -rf /var/lib/apt/lists/*

# ===== Set Working Dir =====
WORKDIR /app

# ===== Copy Project =====
COPY . .

# ===== Install Dependencies =====
RUN pip install poetry && poetry install --no-dev

# Jika pakai requirements.txt:
RUN pip install --no-cache-dir -r requirements.txt

# Install YOLO via ultralytics package
RUN pip install ultralytics

# Opsional: install Weights & Biases
RUN pip install wandb

# ===== Ports =====
EXPOSE 8000

# ===== CMD =====
CMD ["python", "main.py"]
