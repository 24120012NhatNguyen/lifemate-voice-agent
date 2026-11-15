# Gói nộp cho vòng private (Docker)

Tài liệu cho BTC và cho đội khi dựng lại bài nộp. Phương pháp: [METHOD.md](METHOD.md). Kết quả kiểm tra Docker: [BASELINE.md](BASELINE.md) mục 10.

## 1. Gói gồm những gì

| Phần | Ở đâu |
| --- | --- |
| Image Docker `lifemate:98`: mã suy luận, hai môi trường Python ghim phiên bản, trọng số | `lifemate_98.tar.gz` (`docker save`), dựng lại bằng `scripts/docker_build.sh` |
| Mã suy luận | `src/lifemate/`, `prompts/`, `configs/docker.yaml` (kế thừa `configs/experiments/98_*`, `91_*`, `90_*`, `base.yaml`), `docker/` |
| Mã huấn luyện | `scripts/finetune_whisper.py` (ASR), `scripts/build_sft.py` (dữ liệu SFT), `scripts/train_sft.py`, `kaggle/run_sft.py`, `scripts/kaggle_pack.sh`, `scripts/merge_lora.py` (agent) |
| Trọng số | `models/phowhisper-large-ft` (fp16), `models/qwen3-4b-sft-full` (bf16), có trong image; sha256 ở mục 5 |
| Mô tả phương pháp | [METHOD.md](METHOD.md) |

`scripts/package_submission.sh` gom các phần trên vào `artifacts/submission/lifemate_98/` (image, mã nguồn ở commit hiện tại, `MANIFEST.txt` có sha256).

## 2. Chạy

```bash
docker load -i lifemate_98.tar.gz
```

```bash
docker run --rm --gpus all --network none -v /đường/dẫn/gói-dữ-liệu:/data:ro -v /đường/dẫn/kết-quả:/result lifemate:98
```

- `/data` là thư mục chứa `private_test/` (cùng cấu trúc với `public_test/` của gói dữ liệu: `<split>.jsonl`, `audio/`, `<split>_environment/`, `zaic_tools/`, `<split>_tools_catalog.json`), hoặc chính thư mục `private_test/`. Có `private_test/` thì chọn nó; không thì chọn split duy nhất khác `train`; đặt `-e SPLIT=...` để chọn tay.
- Kết quả: `/result/submission.jsonl`, đúng định dạng `sample_submission.jsonl`, đã kiểm bằng `zaic_tools --validate` của split. Kèm theo: `runs/` (transcript, vết agent từng mẫu), `predict.log`, `vllm.log`, `timing.json`, `report.json`.
- Không cần mạng, không cần `--ipc=host` hay `--shm-size`. Chạy như root mặc định của image.
- GPU: một card từ 16 GB. vLLM giữ `GPU_UTIL` (mặc định 0,85) bộ nhớ card khi chạy agent. Driver NVIDIA từ 580; GPU trung tâm dữ liệu như A30 chạy được trên driver cũ hơn nhờ thư viện forward compatibility trong ảnh nền (chưa thử).
- Trên RTX 5060 Ti 16 GB, 634 mẫu public mất 42,4 phút (4,0 giây mỗi mẫu kể cả nạp mô hình): ASR 6,1 phút, khởi động vLLM 1,5 phút, agent 34,8 phút. Mỗi mẫu chạy tuần tự: p99 9,0 giây, chậm nhất 10,7. Giới hạn 25 giây mỗi mẫu (tính cả ASR) là lưới an toàn, chưa cắt mẫu nào. Hai lượt ra cùng file từng byte. Chi tiết: BASELINE mục 10.

Biến môi trường (`-e`): `SPLIT`, `CONFIG` (mặc định `configs/docker.yaml`), `LLM_MODEL`, `ASR_MODEL` (đổi trọng số, vd. gắn thêm `-v /models:/models:ro -e LLM_MODEL=/models/xyz`), `GPU_UTIL`, `LIMIT` (chạy thử trên vài mẫu).

## 3. Dựng lại image

Cần Docker với BuildKit và NVIDIA Container Toolkit; lúc dựng cần mạng (tải ảnh nền, Python, các gói ghim).

```bash
scripts/docker_build.sh
```

- Ảnh nền `nvidia/cuda:13.0.3-base-ubuntu24.04`, Python 3.12.14 và uv 0.12.9.
- `.venv`: gói của dự án theo `uv.lock` (torch 2.11.0+cu128, transformers 4.57.6) cho ASR và pipeline.
- `.venv-vllm`: đúng 199 gói của môi trường đã sinh bản nộp public (`docker/requirements-vllm.txt`: vLLM 0.33.0, torch 2.14.0, triton 3.8.0).
- Trọng số vào image qua hai build context `asr` và `llm`, nên `models/` là liên kết cũng được. Lớp trọng số nằm dưới lớp mã: sửa mã thì dựng lại trong vài giây. `VARIANT=slim` dựng image không có trọng số (gắn bằng `-v <models>:/app/models:ro`).

Kiểm tra: hai lượt cả public trong container phải ra cùng file từng byte, rồi đo độ trễ và bộ nhớ (cần khoá GPU chung khi máy có nhiều người dùng):

```bash
scripts/docker_check.sh
```

## 4. Dựng lại mô hình từ đầu

Môi trường huấn luyện, từ gốc repo, gói dữ liệu BTC đặt ở `../data/lifemate` (hoặc `LIFEMATE_DATA`):

```bash
uv sync --extra asr --extra train --extra sft
```

### 4.1. ASR: PhoWhisper-large-ft

Một GPU 16 GB, khoảng 75 phút trên RTX 5060 Ti. Mặc định của script là đúng công thức đã nộp (3 epoch, lr 2e-4, LoRA r=32, batch 8, cộng dồn 2, warmup 0,1, seed 2026), học trên audio của `train_rest`:

```bash
.venv/bin/python scripts/finetune_whisper.py --model vinai/PhoWhisper-large --out models/phowhisper-large-ft
```

Script tự gộp adapter vào checkpoint fp16 và ghi `train_info.json` (WER dev từng epoch: 26,95 → 10,86 → 9,58 → 8,69).

### 4.2. Dữ liệu SFT

Chạy trên CPU, vài giây. Dùng chữ gold (`audio_text`) và call gold của cả 1.545 mẫu train; 3 mẫu bị bỏ vì call gold bị middleware cảnh báo, còn 1.542:

```bash
.venv/bin/python scripts/build_sft.py --subset all --out sft/train_full.jsonl
```

Bản đã huấn luyện có sha256 `a7580ffc7902baec…`. Dựng lại ngày 05/11 bằng mã hiện tại ra cùng 1.542 hội thoại, giống từng tin nhắn; file khác hash chỉ vì có thêm khoá `"asr": false` (thêm vào script sau đó, `train_sft.py` không đọc).

### 4.3. Huấn luyện agent

Bản nộp được huấn luyện trên Kaggle (T4, fp16), seed 0, khoảng 6 giờ 40 phút cho một lượt trên một T4:

```bash
scripts/kaggle_pack.sh sft/kaggle train_full
```

Tải `sft/kaggle/` lên làm dataset Kaggle riêng tư (dữ liệu BTC chỉ dành cho đội). Notebook bật GPU T4 x2 và Internet:

```
!pip -q uninstall -y torchao
!pip -q install "transformers==4.57.6" "peft==0.21.2" accelerate
!python $(find /kaggle/input -name run_sft.py | head -1) full
```

Tải `qwen3-4b-sft-full-lora.zip` ở tab Output, giải nén vào `models/qwen3-4b-sft-full-lora`, rồi gộp trên CPU (khoảng 2 phút, 16 GB RAM):

```bash
.venv/bin/python scripts/merge_lora.py models/qwen3-4b-sft-full-lora models/qwen3-4b-sft-full
```

Cùng công thức trên một GPU 16 GB của mình (khoảng 3 giờ trên RTX 5060 Ti, đỉnh khoảng 11 GB):

```bash
.venv/bin/python scripts/train_sft.py --data sft/train_full.jsonl --out models/qwen3-4b-sft-full --dtype fp16
```

Huấn luyện lại không ra đúng từng bit trọng số: thứ tự cộng dồn trên GPU khác nhau theo card và phiên bản torch của Kaggle không ghim. Cùng dữ liệu và seed, bản bf16 ở máy nhà được 43,17 public so với 44,70 của bản Kaggle (khác trạng thái cuối ở 159/634 mẫu), cỡ một lần đổi seed. Muốn đúng bài nộp thì dùng trọng số trong image.

### 4.4. Chạy lại ngoài Docker

```bash
.venv/bin/lifemate run configs/experiments/29_asr_phowhisper-large-ft_public.yaml
```

```bash
scripts/gpu_job.sh eval models/qwen3-4b-sft-full configs/experiments/98_submit_public_full_fixes.yaml
```

## 5. Trọng số

| Thư mục | Tham số | Kiểu | sha256 |
| --- | --- | --- | --- |
| `phowhisper-large-ft/model.safetensors` | 1.543.304.960 | fp16 | `2964e9d19183eff09d4151ba63c438961d32d4c47bad734530fc9f6eb53f2a88` |
| `qwen3-4b-sft-full/model-00001-of-00002.safetensors` | 4.022.468.096 (cả hai file) | bf16 | `b206f76b3f49eb1311cf632db3dda027e6cc6a79ccfc4fb8250e7abe14b351a4` |
| `qwen3-4b-sft-full/model-00002-of-00002.safetensors` | | | `2df463cb8d7148d368aa6c0d6a9ca567495c2e96269442f57a56bd803fe1aa88` |

Tổng 5.565.773.056 tham số (dưới 6 tỷ), LLM 4,02 tỷ (dưới 5 tỷ). Mô hình gốc: `vinai/PhoWhisper-large` (BSD-3-Clause, 12/2023), `Qwen/Qwen3-4B-Instruct-2507` (Apache-2.0, 07/2025).
