# syntax=docker/dockerfile:1.7
# Image nộp cho vòng private. Dựng bằng scripts/docker_build.sh; BTC chạy đúng lệnh (không mạng):
#   docker run --gpus '"device=0"' -v <dữ liệu>:/data:ro -v <thư mục ra>:/result IMAGE /bin/bash /code/predict.sh
# Mã ở /app, /code là liên kết tới /app. Kiểm như BTC: scripts/btc_sim.sh <image.tar.gz>.
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

# VARIANT=full2: như full, thêm hai adapter LoRA của ASR (build context `adapter1`, `adapter2`) cạnh bản gốc; dùng với
# configs/docker-qwen35-asr2.yaml (một PhoWhisper gốc + hai adapter thay cho hai bản đã gộp).
FROM weights-full AS weights-full2
ARG ADAPTER1_DIR=phowhisper-large-ft-lora
ARG ADAPTER2_DIR=phowhisper-large-ft2-all-lora
COPY --from=adapter1 / models/${ADAPTER1_DIR}/
COPY --from=adapter2 / models/${ADAPTER2_DIR}/
# Thứ hạng app của train cho mồi ASR (scripts/export_app_rank.py): image không mang dữ liệu train. Build context
# `apprank` là thư mục chứa đúng một file app_rank_train.json (chỉ ID app, không có mẫu).
COPY --from=apprank app_rank_train.json models/app_rank_train.json
ENV LIFEMATE_APP_RANK=/app/models/app_rank_train.json

FROM weights-${VARIANT}
# Config bản nộp mặc định của image (predict.py đọc biến CONFIG); đổi lúc dựng bằng --build-arg CONFIG=...
ARG CONFIG=configs/docker.yaml
ENV CONFIG=${CONFIG}
# JupyterLab cho BTC đo thời gian bằng predict_notebook.ipynb (start_jupyter.sh). Venv riêng: không đụng .venv (uv.lock)
# lẫn .venv-vllm. Lớp này đứng trước mã nên sửa mã không phải cài lại.
COPY docker/requirements-jupyter.txt docker/
RUN uv venv --python 3.12.14 .venv-jupyter \
    && uv pip install --python .venv-jupyter/bin/python -r docker/requirements-jupyter.txt
ENV PATH=/app/.venv-jupyter/bin:${PATH}
# Dữ liệu SFT của LLM (build context `trainingdata`, xem training_code/README.md); base model không vào image.
COPY --from=trainingdata / training_code/data/
COPY README.md LICENSE ./
COPY src src
RUN uv sync --frozen --python 3.12.14 --extra asr
COPY prompts prompts
COPY configs configs
COPY scripts scripts
COPY kaggle kaggle
COPY docker docker
COPY training_code training_code
COPY docs docs
COPY predict.py predict.sh start_jupyter.sh predict_notebook.ipynb requirements.txt ./
# BTC đọc mã ở /code: /code là liên kết tới /app (venv và trọng số giữ đúng đường dẫn lúc dựng).
RUN chmod +x predict.sh start_jupyter.sh docker/predict.sh scripts/*.sh \
    && ln -s /app /code \
    && .venv-jupyter/bin/python -c "import json; json.load(open('predict_notebook.ipynb'))"
WORKDIR /code
# Không ENTRYPOINT: lệnh chấm `docker run ... IMAGE /bin/bash /code/predict.sh` chạy đúng nguyên văn, `docker run IMAGE`
# chạy CMD (cùng lệnh), `IMAGE /bin/bash /code/start_jupyter.sh` mở JupyterLab.
ENTRYPOINT []
CMD ["/bin/bash", "/code/predict.sh"]
