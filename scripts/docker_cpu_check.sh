#!/usr/bin/env bash
# Kiểm một image nộp chỉ bằng CPU: không --gpus, không mạng, không dữ liệu, KHÔNG khởi động vLLM (LM-50, phần B.1).
# Xem cú pháp, đường dẫn, file có đủ trong image, config nạp được, bảng tham số, chat template khớp tokenizer, và cờ
# vLLM mà docker/predict.sh sẽ dùng có trong bản vLLM của image. Khoảng một phút.
#   scripts/docker_cpu_check.sh                    image lifemate:q35
#   IMAGE=lifemate:98s scripts/docker_cpu_check.sh
# Mã thoát 0 khi mọi dòng là ĐẠT. Đây là bước 0 của ngày kiểm Docker; các bước cần GPU nằm ở docs/QWEN35_DOCKER.md.
set -uo pipefail
IMAGE=${IMAGE:-lifemate:q35}
docker run --rm --network none -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 --entrypoint bash "$IMAGE" -c '
set -o pipefail
fail=0
ok() { echo "ĐẠT   $*"; }
bad() { echo "HỎNG  $*"; fail=1; }
chk() { if "${@:2}" > /tmp/chk.log 2>&1; then ok "$1"; else bad "$1: $(tail -2 /tmp/chk.log | tr "\n" " " | cut -c1-240)"; fi; }
echo "CONFIG=$CONFIG LLM_MODEL=$LLM_MODEL ASR_MODEL=$ASR_MODEL"
for f in "$CONFIG" docker/predict.sh scripts/serve_vllm.sh scripts/docker_report.py scripts/asr_latency.py prompts/react_json.md \
         "$LLM_MODEL/config.json" "$LLM_MODEL/tokenizer.json" "$ASR_MODEL/config.json"; do
  if [ -s "$f" ]; then ok "có $f"; else bad "thiếu $f"; fi
done
ls "$LLM_MODEL"/*.safetensors "$ASR_MODEL"/*.safetensors > /dev/null 2>&1 && ok "có trọng số safetensors của LLM và ASR" || bad "thiếu safetensors"
chk "cú pháp docker/predict.sh" bash -n docker/predict.sh
chk "cú pháp scripts/serve_vllm.sh" bash -n scripts/serve_vllm.sh
command -v gcc > /dev/null && command -v curl > /dev/null && ok "có gcc (Triton biên dịch lúc chạy) và curl" || bad "thiếu gcc hoặc curl"
if grep -qs "\"model_type\": *\"qwen3_5_text\"" "$LLM_MODEL/config.json"; then
  ok "LLM là Qwen3.5 chỉ văn bản: predict.sh sẽ chạy --max-num-seqs 1, --gdn-prefill-backend triton, không DETERMINISTIC"
  [ -s prompts/qwen35_scaffold.jinja ] && ok "có prompts/qwen35_scaffold.jinja" || bad "thiếu prompts/qwen35_scaffold.jinja"
  # đọc mã thay cho `vllm serve --help`: không có GPU thì vLLM không đoán được thiết bị và --help thoát lỗi
  chk "vLLM có cờ --gdn-prefill-backend" grep -q gdn_prefill_backend .venv-vllm/lib/python3.12/site-packages/vllm/engine/arg_utils.py
  chk "chat template khung think rỗng khớp tokenizer của model" .venv-vllm/bin/python -c "
import os
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(os.environ[\"LLM_MODEL\"])
t = open(\"prompts/qwen35_scaffold.jinja\").read()
m = [{\"role\": \"system\", \"content\": \"S\"}, {\"role\": \"user\", \"content\": \"U\"}]
s = tok.apply_chat_template(m, chat_template=t, tokenize=False, add_generation_prompt=True)
assert s.endswith(\"<|im_start|>assistant\n<think>\n\n</think>\n\n\"), s[-60:]
assert s == tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
"
else
  ok "LLM không phải Qwen3.5: predict.sh chạy như bản Qwen3 (DETERMINISTIC=1)"
fi
for d in models/*-lora; do
  [ -d "$d" ] || continue
  [ -s "$d/adapter_config.json" ] && [ -s "$d/adapter_model.safetensors" ] && ok "adapter $d" || bad "adapter $d thiếu file"
done
if [ -n "${LIFEMATE_APP_RANK:-}" ]; then
  chk "LIFEMATE_APP_RANK đọc được ($LIFEMATE_APP_RANK)" .venv/bin/python -c "import json, os; d = json.load(open(os.environ[\"LIFEMATE_APP_RANK\"])); assert isinstance(d, dict) and d; print(len(d))"
fi
chk "config nạp được, tuần tự, có time_budget" .venv/bin/python -c "
import os
from lifemate.config import load
c = load(os.environ[\"CONFIG\"], [\"workers=1\", \"agent.time_budget=25\"])
assert c[\"workers\"] == 1 and c[\"agent\"][\"time_budget\"] and c[\"agent\"][\"llm\"][\"temperature\"] == 0.0, c
print(c[\"workers\"], c[\"agent\"][\"time_budget\"], c[\"asr\"])
"
.venv/bin/python scripts/docker_report.py --params-only --models "$ASR_MODEL" "$LLM_MODEL" $(ls -d models/*-lora 2>/dev/null) || fail=1
.venv-vllm/bin/python -c "import importlib.metadata as m; print(\"vllm\", m.version(\"vllm\"), \"torch\", m.version(\"torch\"), \"transformers\", m.version(\"transformers\"), \"triton\", m.version(\"triton\"))"
[ $fail = 0 ] && echo "KIỂM CPU: ĐẠT" || echo "KIỂM CPU: CÓ MỤC HỎNG"
exit $fail'
