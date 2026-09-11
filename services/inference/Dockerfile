# Hugging Face Space (Docker SDK). Spaces expect the app to listen on 7860.
FROM python:3.13-slim

WORKDIR /app

# opencv needs these at runtime even with opencv-python-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Matches the checkpoint this project trained with --base-filters 16
# (see services/inference/training/train_landmark_regression.py). Override
# via Space secrets if you deploy a checkpoint trained with a different value.
ENV LANDMARK_BASE_FILTERS=16
ENV PORT=7860

EXPOSE 7860

CMD python download_checkpoints.py && \
    . /tmp/checkpoint_env.sh 2>/dev/null; \
    uvicorn app:app --host 0.0.0.0 --port ${PORT}
