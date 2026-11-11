# CPU: docker build -t speech-age-gender-api .
# CUDA: docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 -t speech-age-gender-api:cuda .
FROM python:3.12-slim AS runtime

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SPEECH_MODEL_DIR=/models \
    SPEECH_DEVICE=cpu \
    SPEECH_CPU_THREADS=4 \
    HF_HOME=/home/app/.cache/huggingface

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 app && useradd --uid 10001 --gid app --create-home app \
    && mkdir /models && chown app:app /models

COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY scripts/ ./scripts/
# Install the selected PyTorch wheel first so the CUDA/CPU choice is explicit.
RUN pip install torch==2.10.0 --index-url ${TORCH_INDEX_URL} \
    && pip install .

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).close()"
CMD ["speech-age-gender", "serve", "--host", "0.0.0.0", "--port", "8000"]
