# syntax=docker/dockerfile:1.7
# Image nộp cho vòng private (nấc 6). Dựng bằng scripts/docker_build.sh, chạy tắt mạng:
#   docker run --rm --gpus all --network none -v <gói dữ liệu>:/data:ro -v <thư mục ra>:/result lifemate:98
# Hai môi trường Python, đúng như lúc làm bản nộp public, để image ra cùng kết quả:
#   .venv       pipeline, ASR (torch 2.11 cu128, transformers 4.57.6), ghim bằng uv.lock
#   .venv-vllm  vLLM 0.33.0 (torch 2.14 cu130), ghim bằng docker/requirements-vllm.txt
# VARIANT=full (mặc định) chép trọng số vào image từ hai build context tên `asr` và `llm`; VARIANT=slim không có
# trọng số, gắn lúc chạy bằng -v <thư mục models>:/app/models:ro. Trọng số nằm dưới lớp mã, nên sửa mã chỉ dựng lại
# vài lớp nhỏ.
ARG VARIANT=full

FROM nvidia/cuda:13.0.3-base-ubuntu24.04 AS env
ENV DEBIAN_FRONTEND=noninteractive
# gcc và libc6-dev: Triton biên dịch module C nhỏ lúc chạy; curl: chờ vLLM lên
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.12.9 /uv /uvx /usr/local/bin/
ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_NO_CACHE=1 \
    NVIDIA_DISABLE_REQUIRE=1 NVIDIA_DRIVER_CAPABILITIES=compute,utility
WORKDIR /app
RUN uv python install 3.12.14
COPY docker/requirements-vllm.txt docker/
RUN uv venv --python 3.12.14 .venv-vllm \
    && uv pip install --python .venv-vllm/bin/python -r docker/requirements-vllm.txt
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --python 3.12.14 --no-install-project --extra asr

FROM env AS weights-slim
RUN mkdir -p models

FROM env AS weights-full
ARG ASR_DIR=phowhisper-large-ft
ARG LLM_DIR=qwen3-4b-sft-full
COPY --from=asr / models/${ASR_DIR}/
COPY --from=llm / models/${LLM_DIR}/
ENV ASR_MODEL=models/${ASR_DIR} LLM_MODEL=models/${LLM_DIR}

FROM weights-${VARIANT}
# Config bản nộp mặc định của image (docker/predict.sh đọc biến CONFIG); đổi lúc dựng bằng --build-arg CONFIG=...
# (scripts/docker_build.sh: CONFIG=configs/docker-nbest.yaml) hoặc lúc chạy bằng -e CONFIG=...
ARG CONFIG=configs/docker.yaml
ENV CONFIG=${CONFIG}
COPY README.md LICENSE ./
COPY src src
RUN uv sync --frozen --python 3.12.14 --extra asr
COPY prompts prompts
COPY configs configs
COPY scripts scripts
COPY kaggle kaggle
COPY docker docker
RUN chmod +x docker/predict.sh scripts/*.sh
ENTRYPOINT ["/app/docker/predict.sh"]
