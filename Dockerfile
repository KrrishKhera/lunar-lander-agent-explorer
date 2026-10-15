FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SDL_VIDEODRIVER=dummy \
    SDL_AUDIODRIVER=dummy

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential swig && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN python -m pip install --no-cache-dir setuptools wheel && \
    python -m pip install --no-cache-dir --no-build-isolation -r requirements.txt
COPY app.py .
COPY lander ./lander
COPY models/dqn_weights.npz ./models/dqn_weights.npz
COPY results ./results
COPY assets/evaluation_seed_1000.mp4 ./assets/evaluation_seed_1000.mp4

EXPOSE 8080
CMD ["sh", "-c", "streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-8080} --server.headless=true --browser.gatherUsageStats=false"]
