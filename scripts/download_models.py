"""Tải trọng số các mô hình dùng trong configs/experiments vào cache HuggingFace, có thử lại khi mạng treo.

    .venv/bin/python scripts/download_models.py [repo_id ...]

Nếu một file lớn đứng yên (gặp với vinai/PhoWhisper-large qua backend xet), chạy lại với HF_HUB_DISABLE_XET=1.
"""
import os
import sys
import time

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
from huggingface_hub import HfApi, snapshot_download  # noqa: E402

DEFAULT = [
    "openai/whisper-large-v3-turbo",                 # MIT
    "Qualcomm-AI-Research/PhoASR-whisper-small",     # BSD-3-Clause-Clear + Qualcomm RAI license
    "khanhld/chunkformer-ctc-large-vie",             # CC BY-NC 4.0: chỉ thử nghiệm
    "mobiuslabsgmbh/faster-whisper-large-v3-turbo",  # MIT (bản CTranslate2)
    "Qwen/Qwen3-4B-Instruct-2507",                   # Apache-2.0
    "vinai/PhoWhisper-large",                        # BSD-3-Clause
]
SKIP = ["*.msgpack", "*.h5", "*.ot", "flax_model*", "tf_model*", "*.onnx", "onnx/*", "*.DS_Store"]

api = HfApi()
for repo in sys.argv[1:] or DEFAULT:
    files = [f.rfilename for f in api.model_info(repo).siblings]
    ignore = SKIP + (["*.bin"] if any(f.endswith(".safetensors") for f in files) else [])
    for attempt in range(1, 21):
        try:
            path = snapshot_download(repo, ignore_patterns=ignore)
            print(f"ok {repo} -> {path}", flush=True)
            break
        except Exception as e:  # mạng treo hoặc rớt: thử lại, phần đã tải được nối tiếp
            print(f"retry {attempt} {repo}: {type(e).__name__}: {str(e)[:120]}", flush=True)
            time.sleep(5)
    else:
        print(f"FAILED {repo}", flush=True)
