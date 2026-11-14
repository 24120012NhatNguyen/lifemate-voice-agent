# Tái lập bài nộp LifeMate (LM-57, 13/11/2026)

Nền suy luận: `f1f0220` trên `max/q35-kaggle-docker`, cấu hình
`configs/docker-final.yaml`, image đang được bàn giao là `lifemate:final-98f`.
Tài liệu này ghi công thức 98f: **1.542 mẫu train + 600 synth + 2.991 D1 +
948 bản chép ASR theo fold = 6.081 hội thoại SFT**. Tham số lấy từ
`train_info.json`, `train_args.json`, D1 `MANIFEST.json` và `soup.json` cục bộ.
Không suy ra model đang ở trong một tag Docker chỉ từ tên tag: Pro cần đối chiếu
biên nhận dựng image trước khi chốt model/seed của bài nộp.

Các lệnh GPU, Kaggle, teacher và Docker dưới đây là hướng dẫn tái lập, **không
được chạy trong nhiệm vụ CPU LM-57**. Thời gian là ước lượng trừ khi ghi “theo log”.
Huấn luyện lại không bảo đảm trọng số trùng từng bit; dữ liệu teacher cũng không
tất định. Gói dữ liệu đã lọc và hash là đầu vào chuẩn cho việc tái lập huấn luyện.

## 1. Đầu vào, môi trường và mã nguồn

BTC cung cấp gói LifeMate v3.2.0. Tạo thư mục làm việc chỉ chứa `VERSION` và
`train/` (JSONL, catalog, audio, environment và `zaic_tools` của train), đặt:

```bash
export LIFEMATE_DATA=/duong/dan/lifemate-train-only
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PWD/src"
uv sync --frozen --extra asr --extra train --extra sft --extra dev
```

Lệnh `uv sync` là bước chuẩn bị môi trường; có thể cần mạng/cache đã cấp.
Môi trường ASR/SFT-data dùng Python 3.12, transformers 4.57.6, torch 2.11.0,
peft 0.21.2, accelerate 1.15.0 theo môi trường đã dùng. Môi trường Qwen3.5/vLLM
riêng dùng transformers 5.20.2, torch 2.14.0; train thêm peft 0.21.2,
accelerate 1.15.0, fla-core và flash-linear-attention 0.6.0. Chi tiết suy luận
ghim trong `docker/requirements-vllm.txt`. Kaggle giữ torch của ảnh Kaggle,
runner ghi phiên bản thật trong `q35_summary.log`; T4 dùng fp16.
Chuẩn bị môi trường/model nền: khoảng 10–30 phút nếu phải tải.

`scripts/pack_handin.sh` xuất ba archive bằng `git archive`, giải nén ở cùng
một thư mục cha sẽ có:

| Thư mục | Mã và công dụng |
| --- | --- |
| `lifemate/` | HEAD nhánh handin, mã train/inference cuối và bộ sinh sửa đọc catalog |
| `teacher/` | `4d7a232799df0716153a894950d1077517b29a6e`, đúng mã D1 được MANIFEST ghi nhận |
| `asr-fold/` | `16e6402a92a6b57f84f4bb2faac0f704e5801f18`, đúng mã chia fold, chép và dựng SFT lỗi |

Hai archive lịch sử giữ mã ở thời điểm tạo dữ liệu, gồm cả module synth cũ;
chỉ chạy đường teacher/fold dưới đây trong thư mục train-only. Bộ sinh synth
để tái lập là bản sửa trong `lifemate/`, không dùng bộ sinh cũ trong archive.

## 2. Fine-tune PhoWhisper

Đầu vào: model `vinai/PhoWhisper-large`, 5.036 lượt nói của toàn bộ train
(lượt cuối và lịch sử). Mọi lệnh dưới đây chạy từ `lifemate/` với môi trường ASR.
LoRA r=32, alpha 64, attention và FFN của encoder/decoder; lr 2e-4, batch 8,
cộng dồn 2, warmup 0,1, seed 2026.

```bash
.venv/bin/python scripts/finetune_whisper.py \
  --model vinai/PhoWhisper-large --data all --epochs 3 --dtype float16 \
  --seed 2026 --workers 2 --no-merge --save-every-epoch \
  --out models/phowhisper-large-ft-all

.venv/bin/python scripts/finetune_whisper.py \
  --model vinai/PhoWhisper-large --data all --epochs 6 --dtype float16 \
  --seed 2026 --workers 2 --speed 0.9,1.0,1.1 \
  --hint profile_history_apps --hint-prob 0.5 --max-prompt-tokens 200 \
  --no-merge --save-every-epoch --out models/phowhisper-large-ft2-all
```

Đầu ra mỗi lượt: `adapter/`, adapter mỗi epoch, `train_info.json`.
`ft-all` lấy epoch 3, `ft2-all` lấy epoch 6; không chọn epoch bằng public.
Theo log Kaggle, `ft-all` mất 249 phút; `ft2-all` ước 6–10 giờ/T4,
một GPU 16 GB mới hơn ước 2–5 giờ. Kaggle chạy mỗi model trên một T4,
hai model có thể chạy riêng trên hai card; `kaggle/run_asr.py` có công thức `ft2-all`.
Mồi chỉ lấy tên từ hồ sơ, lịch sử và catalog train.

Gộp adapter trên CPU (2–5 phút/model, khoảng 13 GB RAM):

```bash
.venv/bin/python scripts/finetune_whisper.py \
  --merge models/phowhisper-large-ft-all/adapter --out models/phowhisper-large-ft-all
.venv/bin/python scripts/finetune_whisper.py \
  --merge models/phowhisper-large-ft2-all/adapter --out models/phowhisper-large-ft2-all
```

Image cuối dùng **một Whisper gốc + hai adapter**, hai lượt nghe nối tiếp.
Image `final-98f` dùng adapter tách lại từ model gộp bằng SVD; để dựng đúng
cách đó, chạy CPU (ước 5–20 phút/model):

```bash
.venv/bin/python scripts/whisper_lora_extract.py models/phowhisper-large-ft-all \
  --out models/phowhisper-large-ft-all-lora --save-base models/phowhisper-large-base --check
.venv/bin/python scripts/whisper_lora_extract.py models/phowhisper-large-ft2-all \
  --out models/phowhisper-large-ft2-all-lora --check
```

SVD từ bản gộp fp16 không phục hồi adapter train đúng từng bit. Muốn khớp image
đã kiểm phải dùng chính trọng số/adapter của image và biên nhận hash của Pro.

## 3. Dữ liệu synth: seed và thứ tự duyệt

`real_app_ids()` chỉ đọc catalog train. Không cần có thư mục public để sinh.
File cũ `train_synth.jsonl` dùng bảng kịch bản v1 (trước `pick_detail`), seed
2026; phần synth của file full dùng bảng v2, cùng seed. Bộ sinh cũ duyệt `set`,
nên seed Python `random` chưa đủ để dựng lại: cần thứ tự duyệt cũ. Đã đưa cơ chế
phát lại của LM-48 vào nhánh này; metadata chỉ chứa tên trường, thứ tự và hash,
không chứa mẫu BTC hay hội thoại sinh ra.

Chạy từ `lifemate/`, CPU, khoảng 10–30 giây/file:

```bash
.venv/bin/python scripts/build_sft.py --seed 2026 \
  --synth-replay docs/repro/synth_v1_orders.json --out sft/train_synth.jsonl
.venv/bin/python scripts/build_sft.py --subset all --seed 2026 \
  --synth-replay docs/repro/synth_full_v2_orders.json --out sft/train_full_synth.jsonl
```

Đầu vào: `train.jsonl`, catalog, môi trường train và metadata thứ tự.
Đầu ra: 1.618 dòng (1.018 thật + 600 synth) và 2.142 dòng
(1.542 thật + 600 synth). LM-57 đã đối chiếu byte với worktree pro-threshold
chỉ đọc; hash cả file thứ nhất và phần đầu 2.142 dòng full:

```text
b251018b74a93b1a82019d3fd3d3559898bb54f6736122458b34b4ffb939facb  train_synth.jsonl
eb9ed4d76087c20e96277805aa4e998c1d8e23368538a7ac6bc4f99c037be067  train_full_synth.jsonl
```

`scripts/synth_orders.py find --ref FILE --seed 2026 --kinds v1|v2 --subset
train_rest|all --out ORDERS.json` có thể tự tìm lại metadata (CPU vài giây).
`check --ref FILE --replay ORDERS.json --subset ...` kiểm 600 dòng synth.
`--synth 600` sinh mới theo thứ tự chữ cái, tất định giữa các tiến trình nhưng
không phải cách phục hồi dữ liệu cũ đã train.

## 4. D1 do teacher viết và build_sft

D1 là câu diễn đạt lại của mẫu train: giữ lịch sử, state, quỹ đạo gold; chỉ đổi
câu nói cuối. Gói này dùng các mô hình **Claude (Anthropic)** viết và Claude
giải lại độc lập để lọc. Quy ước khai nguồn các họ teacher khác trong dự án:
**GPT (OpenAI)**, **Gemini (Google)**; không mặc định hai họ đó đã sinh D1 này.
Giữ nguyên MANIFEST nguồn; tài liệu không khai tên phiên bản teacher nội bộ.

Chạy trong `teacher/` với `PYTHONPATH="$PWD/src"`, đầu vào chỉ train,
đầu ra nằm ngoài checkout:

```bash
ln -s ../lifemate/.venv .venv  # trong teacher/, dùng môi trường ASR đã chuẩn bị
.venv/bin/python scripts/teacher_synth.py paraphrase --hosts all \
  --variants 3 --keep 2 --examples 3 --attempts 2 --seed 2026 --backend batch \
  --work ../teacher-work/paraphrase --out ../teacher-output/paraphrase
```

Backend batch ghi prompt rồi dừng. Người vận hành cho teacher trả lời vào
`responses/` theo định dạng trong prompt, chạy lại đúng lệnh để nhập và lọc.
Không tự gọi teacher/API trong LM-57. Lọc gồm kiểm tĩnh, replay gold và solver
độc lập đồng ý state hash, lệnh ghi, câu hỏi lại và kiểu phản ứng.
Đầu ra chuẩn: 2.991 dòng D1 all, 1.949 train_rest trong `.samples.jsonl`,
`.sft.jsonl` và MANIFEST. Thời gian ước 2–8 giờ tùy backend/đồng thời;
không bảo đảm chạy teacher lại ra cùng byte. Dùng file đã lọc trong gói nộp.
Mã `export.py` gọi `build_sft.conversation` để dựng message đúng prompt agent.

Trở lại `lifemate/`, ghép dữ liệu trên CPU (vài giây):

```bash
cat sft/train_full_synth.jsonl ../data/d1/paraphrase_all.sft.jsonl > sft/train_full_para_v1.jsonl
```

Đầu ra 5.133 dòng, SHA-256
`d8363b21c7124f37af26971999947f1be2df9b581567e9240325bd11737031d4`.
D2, D1 mở rộng, TTS, simalt và các augmentation khác không thuộc công thức
6.081 dòng này; phải đổi danh sách hash khi quản lý chọn công thức khác.

## 5. ASR năm fold để tạo bản lỗi cho SFT

Chạy từ `asr-fold/` (archive `16e6402`), với môi trường ASR và
`PYTHONPATH="$PWD/src"`. Chia **train_rest** 1.019 mẫu bằng
`SHA1("asr-fold:" + id) mod 5`; mỗi model không học fold mà nó chép,
kể cả audio lịch sử. Lệnh một GPU 16 GB (ước 6–10 giờ cho 5 fold):

```bash
ln -s ../lifemate/.venv .venv  # trong asr-fold/
N=5 KEEP_MODELS=1 GPU_LOCK=/duong/dan/khoa-chung/gpu.lock scripts/asr_folds.sh
.venv/bin/python scripts/asr_fold_merge.py --folds 5
.venv/bin/python scripts/build_sft.py \
  --asr-config configs/experiments/181_asr_folds.yaml --asr-frac 1 \
  --asr-keep-entities-nbest --asr-only --out sft/asrfold_train_rest.jsonl
```

Kaggle tương đương: `kaggle/run_asr_folds.py --folds 0,1,2,3,4 --n 5`
trong kernel có đủ gói ASR và train-only input, chia fold giữa 2 T4
(ước 7–12 giờ; có thể chia thành các kernel fold 0,1 / 2,3 / 4).
Chép cache `cache/asr/train/` về checkout `asr-fold/`, rồi gộp như trên.
Mỗi fold: LoRA r32, 3 epoch, lr 2e-4, seed 2026, speed 1.0, không mồi;
chép beam 4, N-best 4, cả lịch sử. GPU PC phải giữ khóa chung.

Đầu ra: cache của config 181 và **948 dòng** SFT lỗi, 70 mẫu bỏ vì mất thực thể
ở mọi phương án N-best, 1 mẫu bỏ vì middleware cảnh báo. File chuẩn đã được
đóng gói; dùng nó nếu cần đúng dữ liệu 98f, vì train ASR lại có thể đổi bản chép.
Trở lại `lifemate/`:

```bash
cat sft/train_full_para_v1.jsonl ../data/fold/asrfold_train_rest.jsonl > sft/train_full_para_v1_asrfold.jsonl
```

Đầu ra **6.081 dòng**, SHA-256
`7a42b39bf9f98ffc1efcd65fa846aa02f4ecfc79772c52973ffd076fa4d03c24`.
Toàn bộ hash, số dòng, dung lượng và nguồn ở `docs/handin_data.json`;
LM-57 đã kiểm file cuối bằng phép nối từng byte.

## 6. Qwen3.5-4B SFT trên 16 GB hoặc Kaggle T4 ×2

Từ `lifemate/`, model nền là snapshot `Qwen/Qwen3.5-4B` revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, đã có trên máy/cache:

```bash
.venv-vllm/bin/python scripts/export_text_only.py --out models/qwen35-4b-text
```

CPU 2–5 phút, đầu ra 4.205.751.296 tham số văn bản, bỏ vision và MTP,
kèm `export_receipt.json`. Chi tiết: [QWEN35_TEXT_ONLY.md](QWEN35_TEXT_ONLY.md).

Mỗi seed dùng 1 epoch, lr 2e-4, LoRA r16/alpha32/dropout0, batch hiệu dụng 16,
warmup 0,05 cosine, max_len 12.288. Loss chỉ trên nội dung assistant và im_end;
khung think rỗng không tính loss. Mỗi dòng là một hội thoại nhiều lượt.
GPU 16 GB (seed 0 hoặc 1, bf16; ước 8–12 giờ/seed, đỉnh khoảng 12 GB):

```bash
.venv-vllm/bin/python scripts/train_sft_q35r.py --base models/qwen35-4b-text \
  --data sft/train_full_para_v1_asrfold.jsonl --out models/qwen35-4b-sft-full-para-v1-asrfold \
  --epochs 1 --lr 2e-4 --rank 16 --alpha 32 --dropout 0 --grad-accum 16 \
  --warmup 0.05 --max-len 12288 --dtype bf16 --seed 0 \
  --deadline 120 --ckpt-every 30 --resume auto
```

Giữ khóa GPU chung quanh mỗi chặng. Mã thoát 3 nghĩa là checkpoint đã lưu;
chạy lại đúng lệnh đến mã 0. Đầu ra: `<out>-lora`, `<out>` gộp và `train_args.json`.

Kaggle 2×T4, pack cục bộ (không cần credential), tải thư mục này thành dataset
riêng tư trong tài khoản được cấp, gắn vào notebook GPU T4 ×2:

```bash
scripts/kaggle_q35.sh pack full-asrfold-s2 sft/train_full_para_v1_asrfold.jsonl artifacts/kaggle-s2
```

Trong kernel (chọn đúng đường dẫn input, môi trường đã được cấp quyền tải gói/model):

```bash
python /kaggle/input/TEN-DATASET/run_sft.py --ddp --deadline 640 \
  full-asrfold-s2=train_full_para_v1_asrfold.jsonl -- --seed 2
```

Runner cài môi trường Qwen3.5, xuất text-only, dùng DDP 2 rank, fp16 + GradScaler;
batch hiệu dụng vẫn 16, không phải 16 mỗi GPU. Ước 12–16 giờ/seed full,
thường cần phiên chạy tiếp. Đưa checkpoint phẳng từ output phiên trước vào input,
thêm `--require-checkpoint --require-step N` trước tên thí nghiệm để dừng sớm nếu
dataset chưa có đúng checkpoint. Đầu ra `.zip` adapter hoặc checkpoint,
`q35_summary.log`; gộp adapter sau khi tải về, CPU 2–5 phút:

```bash
.venv-vllm/bin/python scripts/merge_lora.py models/qwen35-4b-sft-full-asrfold-s2-lora models/qwen35-4b-sft-full-asrfold-s2 \
  --base models/qwen35-4b-text
```

Lặp seed 2–6 nếu tái lập ứng viên soup 7 seed; mỗi lượt phải có tài nguyên riêng
được duyệt. Các helper `push/auto` phụ thuộc `tools-pro` của Pro; dùng cách pack và
runner trực tiếp ở trên để gói nộp không phụ thuộc script/credential riêng.

## 7. Soup nhiều seed và dựng Docker

Ứng viên có `soup.json` cục bộ ngày 13/11 là trung bình đều seed 0–6 của cùng
file 6.081 dòng. Hai seed 0,1 dùng bf16 GPU nhà, seed 2–6 dùng fp16 Kaggle.
Đây là ứng viên; Pro chốt model thực sự gắn vào image. Chạy CPU (ước 5–15 phút,
cần khoảng 40–64 GB RAM vì script giữ tensor nguồn và kết quả):

```bash
.venv-vllm/bin/python scripts/soup.py models/qwen35-4b-soup-98f-7 \
  models/qwen35-4b-sft-full-para-v1-asrfold \
  models/qwen35-4b-sft-full-para-v1-asrfold-s1 \
  models/qwen35-4b-sft-full-asrfold-s{2,3,4,5,6}
```

Đầu vào: model đã gộp LoRA cùng model nền; tính trung bình fp32, trả về dtype
gốc. Đầu ra: `model.safetensors`, tokenizer/config, `soup.json` ghi nguồn/hệ số.
`train_args.json` được chép từ model đầu; không dùng nó làm biên nhận cho cả soup.

Dựng app rank chỉ từ train và Docker (CPU 5–15 phút khi có cache; lần đầu
tải ảnh/gói có thể 30–60 phút):

```bash
.venv/bin/python scripts/export_app_rank.py models/app_rank_train.json
ASR=models/phowhisper-large-base \
ADAPTERS="models/phowhisper-large-ft-all-lora models/phowhisper-large-ft2-all-lora" \
APP_RANK=models/app_rank_train.json LLM=models/qwen35-4b-soup-98f-7 \
CONFIG=configs/docker-final.yaml TAG=lifemate:final-98f-handin scripts/docker_build.sh
IMAGE=lifemate:final-98f-handin scripts/docker_cpu_check.sh
```

Image nghe ft-all N-best 4 rồi ft2-all greedy có mồi; agent tuần tự, tối đa
25 giây/mẫu, các guard của 98f và `empty_retry`. Tổng tham số một LLM + một
Whisper gốc + hai adapter: 5.864.399.616. Đầu ra build:
`artifacts/docker/lifemate_final-98f-handin.txt`, image và biên nhận hash.
Đo GPU, độ trễ và hai lượt Docker là việc riêng cần ngân sách; không tuyên bố
đã đo lại chúng trong LM-57. Khi nộp private chạy container offline với input
BTC gắn chỉ đọc và kết quả ghi ra ngoài; xem [QWEN35_DOCKER.md](QWEN35_DOCKER.md).

## 8. Đóng gói và phần quản lý chốt

Pro chạy từ checkout sạch, đã commit. Script chỉ đọc 9 file dẫn xuất được
allowlist/hash, không quét thư mục dữ liệu BTC; không gom audio/model.
Các hội thoại SFT có nội dung dựng từ train được phép có trong gói nộp;
không chép nguyên `train.jsonl`, catalog hay environment BTC vào gói.
MANIFEST D1 giữ nguyên byte, gồm provenance gốc. Thư mục đích phải chưa có.

```bash
scripts/pack_handin.sh /duong/dan/ngoai-git/handin-98f
cd /duong/dan/ngoai-git/handin-98f
sha256sum -c SHA256SUMS
```

Máy khác có thể truyền `--sft-root DIR --d1-root DIR --fold-root DIR`.
Nếu clone thiếu commit lịch sử, fetch nhánh chứa `4d7a232` và `16e6402` trước;
script dừng khi thiếu mã/file/hash, không tải gì tự động. Ước 5–20 giây CPU,
khoảng 291 MB (278 MiB) dữ liệu cộng ba archive mã và MANIFEST. Output chỉ để gửi BTC,
không đẩy dữ liệu lên git. Image/trọng số được quản lý đóng gói riêng.

Kiểm tra CPU: `PYTHONPATH=src LIFEMATE_DATA=... CUDA_VISIBLE_DEVICES= pytest -q`
và `ruff check .`. Các test Qwen3.5 cần môi trường transformers 5.x và tokenizer;
môi trường ASR 4.x tự bỏ qua nhóm đó. Test pack dùng dữ liệu giả, không chạy
script pack trên dữ liệu thật thay Pro.

Audit repo hiện tại: không còn lời gọi đọc public trực tiếp trong synth, train,
viewer hay test schema; chỉ còn config/CLI suy luận, kiểm Docker, đo latency,
chú thích và một test agent giả mang tên split public. Việc bỏ đọc catalog
không tự giải quyết lịch sử thiết kế: `HELDOUT_APPS`, chủ đề synth và một số luật
suy luận đã có tham chiếu public trong quá trình phát triển (được LM-48 ghi nhận).
Không chỉnh các luật suy luận trong LM-57. Quản lý cần duyệt cách khai báo lịch sử
này với BTC, chốt model/seed/hệ số soup và hash image cuối; metadata hiện tại chỉ
cho phép đóng gói công thức 98f 6.081 dòng, sẽ dừng nếu dữ liệu đổi.
