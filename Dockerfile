FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Enable Git LFS
RUN git lfs install

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pull LFS files
RUN git lfs pull

EXPOSE 7860

CMD ["python", "app.py"]