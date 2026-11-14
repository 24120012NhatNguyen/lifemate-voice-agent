# Train Qwen3.5-4B chỉ văn bản trên Kaggle T4 ×2 · LM-50

Trước LM-50, Qwen3.5 chỉ train được ở GPU nhà (RTX 5060 Ti, bf16). Tài liệu này ghi cách train trên Kaggle 2×T4, những gì đã đo, và các bẫy đã gặp. Mọi con số dưới đây là số đo ngày 11/11/2026, trừ chỗ ghi rõ là ước lượng.

## 1. Một lệnh

Chạy trên PC của Pro, trong một worktree có nhánh này (ví dụ `fable-work/lm50`):

```bash
setsid nohup scripts/kaggle_q35.sh auto nbinhdz nbinhdz <tên> <pro-threshold>/sft/<file>.jsonl configs/experiments/<tiền tố config> > auto_<tên>.out 2>&1 &
```

`auto` đẩy kernel `nbinhdz/lifemate-q35-<tên>`, chờ, phiên nào hết hạn thì tự đẩy phiên chạy tiếp từ checkpoint, tải adapter về, gộp LoRA trên CPU thành `models/qwen35-4b-sft-<tên>` (gắn vào `pro-threshold/models/`), rồi đo `<tiền tố>_val-app.yaml` và `<tiền tố>.yaml` qua khoá GPU chung bằng `tools-pro/gpu_eval.sh`. Mốc ghi vào `night.log` với tiền tố `Q35-Kaggle <tên>`. Bỏ tiền tố config thì dừng sau khi gộp.

Từng bước riêng: `push`, `smoke`, `status`, `fetch`, `resume`, `pack` (xem đầu `scripts/kaggle_q35.sh`). Ba tài khoản: `nbinh nhatnguyenhcmusk24`, `hisu hiuinhc`, `nbinhdz nbinhdz` (tên file token, tên người dùng Kaggle). Script không đọc `~/.kaggle`: việc chạm token đi qua `tools-pro/kpush.sh`, `kaggle_fetch.sh` và `kstat` của Pro.

Đối số thêm cho runner đi trong chuỗi thứ năm, sau `--` là đối số của trainer: `push ... "--deadline 300 -- --epochs 2 --seed 1"`. Mặc định `--deadline 640` phút.

## 2. Gói chạy gì trong kernel

`kaggle/run_sft_q35.py` (vào dataset với tên `run_sft.py`, nên `kpush.sh` dùng nguyên):

1. In Python, GPU, driver, đĩa, RAM. Cài `transformers==5.20.2 peft==0.21.2 accelerate fla-core==0.6.0 flash-linear-attention==0.6.0` với ràng buộc giữ nguyên torch của ảnh Kaggle. Không cài được bộ có FLA thì cài bộ không có FLA và ghi cảnh báo.
2. Model nền: tải `Qwen/Qwen3.5-4B` revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` (công khai, 8,7 GiB) vào đĩa tạm, xuất bản chỉ văn bản bằng `scripts/export_text_only.py` (4.205.751.296 tham số, bỏ vision và MTP), xoá bản tải. Nếu input đã gắn một thư mục có `export_receipt.json` thì dùng luôn.
3. Train bằng `scripts/train_sft_q35r.py` qua `torch.distributed.run` 2 rank. T4 không có bf16 thật nên trainer tự chọn fp16 + GradScaler. Checkpoint mỗi 30 phút và khi sắp quá hạn phiên.
4. Xong: `qwen35-4b-sft-<tên>-lora.zip`. Chưa xong: `qwen35-4b-sft-<tên>-ckpt.zip` chứa các file phẳng `ckpt__<tên>__*`; `fetch` giải ra, `resume` đưa chúng vào dataset của phiên sau, runner tự đặt lại vào thư mục checkpoint. Phiên sau có thể ở tài khoản khác và số GPU khác.
5. `q35_summary.log` (JSON): phiên bản thư viện, thời gian từng bước, tok/s, bộ nhớ đỉnh từng GPU, loss, số mẫu bị bỏ, kernel nào đã dùng.

Lưới an toàn cho hạn mức: `--smoke` giết một chặng chạy quá 40 phút; cả phiên có đồng hồ tổng; lượt thường bị giết nếu quá hạn 30 phút mà trainer chưa thoát.

## 3. Đã đo trên Kaggle (smoke 2, `nbinhdz/lifemate-q35-smoke2`, 19 phút)

| Mục | Kaggle 2×T4 | Máy nhà (RTX 5060 Ti) |
|---|---|---|
| Thư viện | Python 3.13, torch 2.11.0+cu130, triton 3.6.0, transformers 5.20.2, peft 0.21.2, FLA 0.6.0 | torch 2.14.0, triton 3.8.0, cùng transformers / peft / FLA |
| Cài thư viện | 76 giây | — |
| Model nền | tải 47 giây, xuất 109 giây; sha256 hai file safetensors trùng `pro-threshold/models/qwen35-4b-text` | — |
| Kernel Gated DeltaNet | FLA (Triton) chạy được trên sm75; không có dòng `chunk_gated_delta_rule … falling back` | FLA |
| Kernel conv | bản PyTorch tham chiếu (`causal_conv1d` không cài) | như vậy |
| Loss bước 1, bước 2 (cùng 16 mẫu đầu, seed 0, lr 2e-4) | 0,2878 và 0,2059 (fp16, 2 rank) | 0,2876 và 0,2058 (fp16, 1 GPU, pilot LM-50); 0,2876 (bf16, FLA); 0,2867 (bf16, kernel tham chiếu) |
| Mẫu loss không hữu hạn | 0 trong 64 mẫu | 0 trong 96 mẫu (pilot fp16) |
| Tốc độ | 163 tok/s ở bước 1 (gồm biên dịch kernel Triton), 257 sau bước 2, 347 cộng dồn sau bước 4 | 890 tok/s (fp16, 6 bước), 721 (bf16) |
| Bộ nhớ đỉnh | 10,8 GiB đã cấp ở rank 0; nvidia-smi 12.197 và 12.037 MiB trên 15.360 | 11,2 GiB (fp16) |
| Dừng rồi chạy tiếp trong phiên | đạt (dừng sau bước 2, chạy tiếp từ bước 2/4) | — |

Tốc độ ổn định trên 2×T4 **ước lượng** khoảng 577 tok/s: chặng chạy tiếp (kernel đã nằm trong cache Triton) đi 124 nghìn token trong 215 giây. Số đo thật của lượt đầy đủ nằm ở mục 5.

## 4. Bẫy đã gặp

**Hết bộ nhớ trên T4 vì SDPA rơi về đường "math".** Smoke 1 chết ngay bước đầu ở cả kernel FLA lẫn kernel tham chiếu: `13,06 GiB đã cấp + xin 1,27 GiB` trên 14,56 GiB. Qwen3.5 có 16 đầu attention, 4 đầu K/V, `head_dim` 256 và trainer không truyền mask, nên `transformers/integrations/sdpa_attention.py` gọi `scaled_dot_product_attention(..., enable_gqa=True)`. T4 (sm75) không có kernel flash, còn kernel mem-efficient không nhận GQA: PyTorch dùng đường math và dựng cả ma trận 16 × L × L (1,27 GiB ở khoảng 6.500 token, vài bản trong backward). Phép dò trong runner ghi lại trên T4: `enable_gqa` → flash không, efficient không; lặp K/V theo nhóm → efficient có. `train_sft_q35r.py` vì thế ép Transformers lặp K/V theo nhóm khi compute capability dưới 8 (`keep_efficient_sdpa`). GPU từ Ampere trở lên, gồm RTX 5060 Ti, không đổi gì.

**Dấu phân cách trong log.** `torchrun` in các hàng `=====` khi một rank thoát mã 3 (dừng có chủ ý). Runner dùng dấu riêng `##### LƯỢT` cho đầu mỗi lượt.

**Liên kết trong thư mục dữ liệu.** `fable-work/data/lifemate/train` là liên kết ra ngoài; container Docker không đi theo được. Xem `scripts/docker_check.sh`, biến `DOCKER_ARGS`.

## 5. So với 161 (cùng dữ liệu, seed, siêu tham số; khác fp16, 2 rank, torch 2.11)

Lượt `nbinhdz/lifemate-q35-synth-k` (`train_synth.jsonl`, 1.618 mẫu, 102 bước, seed 0). Model gộp là `models/qwen35-4b-sft-synth-k`, đo bằng `configs/experiments/161k_qwen35-sft_synth-kaggle{_val-app,}.yaml` (bản sao 161, chỉ đổi model), 8 luồng, `DETERMINISTIC=0`, đêm 11/11. Tiêu chí của brief: lệch 161 không quá 1,5 điểm mỗi tập.

| Tập | 161 (máy nhà, bf16) | 161k (Kaggle 2×T4, fp16) | Hiệu | Ghép cặp từng mẫu (95%) | Tốt hơn / kém hơn |
|---|---:|---:|---:|---|---|
| val_app (301 mẫu) | 46,28 | 46,45 | +0,17 | +0,17 ± 3,97 | 29 / 32 |
| val_random (225 mẫu) | 66,99 | 67,25 | +0,26 | +0,26 ± 4,38 | 22 / 25 |

**Đạt** tiêu chí 1,5 điểm ở cả hai tập. Để so: 161 đo lại lần hai cùng model cho 45,94 / 66,99, tức riêng phép đo đã lệch tới 0,34 điểm. Model 161k đến từ **một phiên liền** 102 bước: 518 tok/s cộng dồn (6,57 triệu token trong 12.880 giây, kể cả biên dịch kernel), loss 0,2878 → 0,0049, 0 mẫu bị bỏ, bộ nhớ đỉnh 11,6 GiB đã cấp và 12.971 / 12.791 MiB theo nvidia-smi. Một lượt như vậy tốn khoảng 3 giờ 40 phút hạn mức, kể cả dựng.

**Checkpoint qua phiên trên Kaggle chưa đạt.** Lượt được chia hai phiên có chủ ý (`--deadline 115`). Phiên 1 dừng đúng ở bước 52 và để lại checkpoint 372 MB; `fetch` và `resume` đưa năm file `ckpt__synth-k__*` vào thư mục đẩy. Nhưng log trainer của phiên 2 ghi `--resume auto: chưa có checkpoint, học từ đầu`: phiên 2 học lại cả 102 bước (218 phút), 112 phút của phiên 1 bỏ đi. Tôi chưa xác định được nguyên nhân; khả năng là kernel bắt đầu khi dataset chưa lên bản có checkpoint (`kpush.sh` chờ dataset tối đa 10 phút rồi vẫn đẩy kernel). Đã có: dừng rồi chạy tiếp trong cùng phiên (smoke 2), và chạy tiếp qua "phiên" khác trên CPU (test). Đã thêm `--require-checkpoint` cho phiên chạy tiếp (không thấy checkpoint thì dừng sau vài phút thay vì học lại); bản vá này chưa chạy trên Kaggle. Tới khi kiểm được, chỉ tin một phiên liền: ở 518 tok/s một phiên 11 giờ chứa khoảng 20 triệu token. Dữ liệu cỡ bản full + D1 (khoảng 21 triệu token theo `q35_full.sh`) không vừa, nên phải kiểm bản vá bằng một lượt thật trước khi train bản cuối trên Kaggle.

Hạn mức nbinhdz LM-50 đã dùng: 11 + 19 + 112 + 218 phút = 6 giờ.

## 6. Kiểm trên CPU

```bash
CUDA_VISIBLE_DEVICES= QWEN35_TOKENIZER=<pro-threshold>/models/qwen35-4b-text <venv có transformers 5.20.2 + peft + pytest>/bin/python -m pytest -q tests/test_kaggle_q35.py
```

Ba test, khoảng 80 giây, model Qwen3.5 hai lớp với tokenizer thật, chạy qua đúng gói do `kaggle_q35.sh pack` dựng: smoke 2 rank; checkpoint phẳng sang phiên khác lệch chạy liền không quá 1e-6; 2 rank lệch 1 rank không quá 1e-5. Không đặt `flash-linear-attention` lên `PYTHONPATH` khi chạy CPU.
