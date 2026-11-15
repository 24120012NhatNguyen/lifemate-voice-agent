# Tái lập bản cuối — nháp LM-99

**Chưa chốt recipe/model cuối.** `SUBMISSION.md` mô tả khung Docker; file này là đường tái lập
mã và các ứng viên. Công thức 98f cũ, lệnh ASR/D1/fold, seed và revision đã có bằng chứng
được giữ nguyên ở `REPRODUCE_98F_ARCHIVE.md`. Không lấy danh sách 98f cũ làm manifest cuối
khi quản lý chọn soup từ nhánh mới. Không lệnh train, tải model, teacher, GPU hoặc Docker nào
trong tài liệu này đã được chạy bởi LM-99.

## Chuẩn bị

Mã train chính: `scripts/train_sft_q35r.py`; runner Kaggle `kaggle/run_sft_q35.py`;
gộp checkpoint `scripts/soup.py` và `scripts/merge_scaled.py`. Nền Qwen3.5 chỉ văn bản
4.205.751.296 tham số, xuất bằng `scripts/export_text_only.py`. Mã/schema/model gốc và
môi trường phải đúng receipt của seed đã chọn; không tải dependency mới trong công việc CPU.

```bash
export LIFEMATE_DATA=/path/to/workspace/data/lifemate
export PYTHONPATH="$PWD/src:$PWD/scripts"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1  # các bước dữ liệu/CPU bên dưới
```

Chỉ train/ được các builder đọc; khi dựng lại bên ngoài máy hiện tại có thể cấp một root
chỉ chứa train và VERSION. Môi trường CPU hiện có là Python 3.12, ASR/pipeline transformers 4.x;
trainer Qwen3.5 cần môi trường 5.x theo runner. `uv.lock` và `docker/requirements-vllm.txt`
là dependency của image, chưa được cài hoặc dựng lại trong LM-99.

## Ma trận dữ liệu ứng viên

Đường dẫn chung hiện tại: `fable-eval/pro-threshold/sft/`; hash đầy đủ ở
`submission_data_candidates.json`. Các số là kiểm kê file thật ngày 15/11, không phải số suy ra
của một model đã chốt. Bản full dùng toàn train, bao gồm phần từng giữ làm val; không dùng
điểm val của model học full làm điểm đo sạch.

| Nhánh | Bản đo train_rest | Bản full | Trạng thái |
|---|---:|---:|---|
| Nền 175 / 98f | 4.515 | 6.081; thêm ASR val: 6.561 | chờ chọn seed/soup |
| D1-TTS v2 | 5.412; thêm fold: 6.360 | 7.935 | chờ chọn seed/soup |
| Guard recovery | 4.774 = 4.515 + 259 | 6.928 = 6.561 + 367 | chờ chọn + xử lý review LM-96 |
| RenamerV2 | 5.644 | 8.201 | chờ chọn seed/soup |
| Synth v3 1.000 | 4.915 | 6.961 | thay 600 synth cũ; chờ chọn + LM-96 |
| Synth v3 400 | 4.315 | 6.361 | thay 600 synth cũ; chờ chọn + LM-96 |
| Calc | 4.515 (123 dòng calc) | 6.561 (199 dòng calc) | chờ chọn + bật agent.calc nếu dùng |

Review LM-96/98 vẫn cần quyết định: nhãn guard mất căn cứ do ASR, dữ liệu chỉ có note đúng,
nhánh synth v3 cần xem các phát hiện chất lượng đã báo, heuristic calc có một bước không cần
thiết về nghĩa. LM-99 chép mã và ghi hash, không lặng lẽ sửa dữ liệu của các seed đang train.

## Bộ sinh synth cũ: chỉ train, cùng byte

`real_app_ids()` đã được sửa train-only trong tổ tiên `5a13031`; không cần gỡ lại. Replay
metadata giữ thứ tự set lịch sử (seed random không đủ). CPU:

```bash
.venv/bin/python scripts/build_sft.py --seed 2026 \
  --synth-replay docs/repro/synth_v1_orders.json --out sft/train_synth.jsonl
.venv/bin/python scripts/build_sft.py --subset all --seed 2026 \
  --synth-replay docs/repro/synth_full_v2_orders.json --out sft/train_full_synth.jsonl
```

Kết quả 1.618 và 2.142 dòng; SHA-256 lần lượt:

```text
b251018b74a93b1a82019d3fd3d3559898bb54f6736122458b34b4ffb939facb
eb9ed4d76087c20e96277805aa4e998c1d8e23368538a7ac6bc4f99c037be067
```

File thứ hai bằng 2.142 dòng đầu `train_full_para_v1.jsonl`; 2.991 D1 ở cuối được teacher
sinh/lọc riêng. Các bước dựng D1/fold và nguồn Claude xem archive 98f. GPT/Gemini chỉ khai
nếu MANIFEST của dữ liệu thực tế có dùng các họ đó.

## Các bộ sinh bổ sung

`docs/repro/final_code_sources.json` giữ path/hash mã tác giả và snapshot trong repo.
Khác biệt khi đưa mã vào: đường dẫn worktree thành repo-relative, tách câu lệnh để đạt Ruff,
bỏ biến/import không dùng. `build_guard_traces.py` thêm `--asr-cache`, dừng nếu cache thiếu
thay vì âm thầm sinh chỉ perturb; `configs/repro-guard-240.yaml` giữ đúng guards nguồn.
Không chạy lại ASR hay teacher trong các lệnh dữ liệu này.

D1-TTS v2 dùng các bản nghe đã tạo trước, giữ nhãn và lọc thực thể theo build_sft:

```bash
.venv/bin/python scripts/build_v2.py tts sft/BASE.jsonl sft/TTS_ASR.jsonl sft/tts_d1_errors.jsonl
# Ghép additive theo recipe đã chốt; không ghi đè file đầu vào.
# ASR của val (chỉ dành cho bản full):
.venv/bin/python scripts/build_v2.py val RUN_VAL_APP RUN_VAL_RANDOM sft/val_asr.jsonl
```

Guard traces (cache export `asr181_train_rest.json` hiện ở `fable-eval/tools-pro/`, phải đưa
vào gói dữ liệu dẫn xuất nếu chọn nhánh này, hash trong báo cáo LM-99):

```bash
.venv/bin/python scripts/build_guard_traces.py --subset train_rest \
  --asr-cache /path/asr181_train_rest.json --out sft/guard_traces_rest.jsonl
.venv/bin/python scripts/build_guard_traces.py --subset all \
  --asr-cache /path/asr181_train_rest.json --out sft/guard_traces_full.jsonl
```

Nối file guard sau baseline tương ứng. Không bỏ `train:false` khi đóng gói.
RenamerV2 giữ dòng gốc và thêm bản đổi tên, seed 2611, ratio 0,25:

```bash
.venv/bin/python scripts/build_rename_v2.py --inp sft/train_para_v1_asrfold.jsonl \
  --out sft/train_para_v1_asrfold_ren2.jsonl
.venv/bin/python scripts/build_rename_v2.py --inp sft/train_full_para_v1_asrfold_val.jsonl \
  --out sft/train_full_para_v1_asrfold_val_ren2.jsonl
```

Synth v3 cần hai cache train đã có: `cache/asr/train/hf_whisper_97e4c4320d.jsonl` và
`gold_2c6032ed1a.jsonl` (nguồn lm51-asr). Không tự sinh cache bằng GPU. CPU:

```bash
.venv/bin/python scripts/synth_v3/gen_v3.py --n 1000 --seed 3003 --out sft/synth_v3.jsonl
.venv/bin/python scripts/synth_v3/finalize_v3.py
# Sau khi cấp tokenizer local models/qwen35-4b-text/tokenizer.json:
.venv/bin/python scripts/synth_v3/check_v3.py
```

LM-99 đã tái sinh 1.000 dòng và metadata đúng byte; SFT SHA-256
`ab5ca8e92f8cef0a8c25ddfcbf53fc1adb633473526c0c7f0fbba06b94065aad`.
Bản 400 phải dùng đúng file/hash trong manifest; không giả định lấy bất kỳ 400 dòng nào.
Finalize thay synth cũ, không cộng cả hai bộ. Không coi phép tái lập byte là xác nhận chất lượng nhãn.

Calc dùng `build_sft` với mode opt-in, không gọi ASR/LLM:

```bash
.venv/bin/python scripts/build_sft.py --config configs/experiments/lm98_calc_data.yaml \
  --from-sft sft/train_para_v1_asrfold.jsonl --out sft/train_para_v1_asrfold_calc.jsonl
.venv/bin/python scripts/build_sft.py --config configs/experiments/lm98_calc_data.yaml \
  --from-sft sft/train_full_para_v1_asrfold_val.jsonl --out sft/train_full_para_v1_asrfold_val_calc.jsonl
```

Thêm `sys.calc` vào prompt mọi dòng; chèn bước tính ở gold/fold khi heuristic xác định.
Source và output, phân bố, từng phép tính nằm trong sidecar `.stats.json`/`.check.json` của LM-98.
Chọn model calc thì bật `agent.calc=true`; cờ cho thêm 4 lượt vào giới hạn ReAct, vẫn chịu 25 giây.

## Train, soup và DPO — chỉ sau khi cấp tài nguyên

Chạy công thức của runner với **file dữ liệu đã chọn**, ghi `train_args.json`, seed, hash dữ liệu,
model revision và dependency versions. Mỗi seed mới có thể có max_len/seed khác: không lấy
mặc định runner thay biên nhận thật. `train_sft_q35r.py` có resume/checkpoint và mask per-message.
Ví dụ gói Kaggle (lệnh chuẩn bị; LM-99 không upload hay train):

```bash
scripts/kaggle_q35.sh pack CANDIDATE sft/SELECTED.jsonl artifacts/kaggle-CANDIDATE
# Trong môi trường được cấp GPU, dùng runner kèm gói và seed đã chốt:
# python run_sft.py --ddp --deadline 640 CANDIDATE=SELECTED.jsonl -- --seed SEED
```

`scripts/soup.py` gộp các checkpoint theo giao diện trong docstring; giữ `soup.json`.
`scripts/merge_scaled.py` hỗ trợ SFT adapter soup fp32 và DPO delta trước một lần cast bf16;
không coi hai phép gộp này là cùng byte. Xem `MERGE_SCALED.md` về thứ tự, hệ số và lineage.
Ứng viên tạm 98f7-DPO α0,25 kế thừa lịch sử LM-81; mã train/dữ liệu DPO riêng nằm trên
`codex/handin-dpo` và phải đưa vào hồ sơ nếu quản lý giữ DPO trong model cuối. Không khai rằng
snapshot CPU này tự tái tạo toàn bộ các seed/DPO chỉ bằng bốn script gộp/train được yêu cầu.

## Kiểm CPU, swap và các bước còn lại

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
.venv/bin/python scripts/check_final_parity.py /tmp/final-parity.json
scripts/docker_swap.sh /path/to/merged-Qwen35 'guards.day_filter=true'
scripts/docker_cpu_host_check.sh
```

Parity dùng mã thật 07e01a4 lấy từ Git và 1.545 mẫu train, phản hồi gold định sẵn, không inference.
Dữ liệu synth tái lập không đổi. Một nhóm test tiny-Qwen3.5 train/resume có thể skip ở môi trường
ASR transformers 4.x; test mask loss thuần CPU vẫn chạy. Thông tin test thực tế ở báo cáo LM-99.

Khi swap hợp lệ, config mới và receipt CPU được ghi; lỗi input/flag/model xảy ra trước khi thay
config. Mặc định mọi cờ mới reset về tắt; build dùng bốn context chính xác và kiểm lại headers.
Không hash toàn payload ở bước CPU này. Chỉ sau quyền dựng mới chạy `scripts/docker_build.sh`,
kiểm image CPU, smoke GPU, latency/VRAM và lặp kết quả theo checklist `SUBMISSION.md`.

`pack_handin.sh` + `handin_data.json` hiện là **allowlist lịch sử 98f 6.081 dòng**, không phải gói
cuối cho mọi nhánh. Trước đóng gói, quản lý chốt recipe/soup, chọn file từ manifest ứng viên,
bổ sung cache/MANIFEST nguồn cần thiết và cập nhật allowlist. Không quét toàn thư mục dữ liệu
BTC để gom gói. Nội dung draft có local paths phục vụ bàn giao; bản gửi BTC cần đổi thành
đường dẫn tương đối trong gói, giữ hash/provenance.
