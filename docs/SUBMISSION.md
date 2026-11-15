# Hồ sơ nộp LifeMate — NHÁP LM-99, 15/11/2026

**Chưa có image cuối. Chờ Pro chốt checkpoint/soup, cờ và dữ liệu sáng 16/11.**
Đây là hồ sơ cho khung Docker Qwen3.5-4B, không phải biên nhận đã dựng hay đã nộp.
Các số Docker của config 98 cũ ở `SUBMISSION_98_ARCHIVE.md` không đại diện bản này.

## Phương pháp và mã

Audio → PhoWhisper-large gốc + adapter ft-all (N-best 4) → thêm bản nghe ft2-all
(greedy, mồi `profile_history_apps`) → Qwen3.5-4B văn bản đã gộp → ReAct JSON/schema,
guards → ToolEnv. Hai ASR chạy nối tiếp, chia sẻ cùng một bản trọng số gốc trong image.
Không dùng soup ASR; kết quả −0,36 dạng public trong brief là lý do quản lý đã loại hướng đó,
không phải phép đo mới của LM-99.

| Thành phần | Mã trong gói |
|---|---|
| Suy luận | `src/lifemate/`, `prompts/`, `configs/docker-final.yaml`, `docker/` |
| Train Qwen3.5 | `scripts/train_sft_q35r.py`, `kaggle/run_sft_q35.py`, `scripts/kaggle_q35.sh` |
| Gộp model | `scripts/soup.py`, `scripts/merge_scaled.py`; chi tiết `docs/MERGE_SCALED.md` |
| ASR | `scripts/finetune_whisper.py`, `scripts/whisper_lora_extract.py`, `scripts/export_app_rank.py` |
| Dữ liệu | `scripts/build_sft.py`, `scripts/build_v2.py`, `scripts/build_guard_traces.py`, `scripts/build_rename_v2.py`, `scripts/synth_v3/`, `scripts/check_calc_sft.py`, `src/lifemate/numbers.py` |
| Tái lập | `docs/REPRODUCE.md`, `docs/repro/final_code_sources.json`, `docs/submission_data_candidates.json` |

Trainer giữ bước assistant có `"train": false` trong ngữ cảnh nhưng mask toàn bộ loss
của body và `<|im_end|>`; không thay token đầu vào. Không có key này thì nhãn như trước.
Mã train/data bổ sung chỉ phục vụ tái lập, không được khởi chạy bởi entrypoint Docker.

## Config và cờ

`configs/docker-final-268.yaml` là snapshot **config.load** của 268 (242 +
`act_after_retry=true`, `stop_scope=any`, `later_stop_min=0.8`, `max_overrides=1`).
`docker-final.yaml` khớp toàn bộ tham số đó ngoài đường dẫn model và biểu diễn ASR
bằng base + adapter. Config raw giữ `workers=8`, `time_budget=null` của 268;
entrypoint Docker đặt `workers=1`, `agent.time_budget=25`. Đây là khác biệt triển khai
có chủ ý, cần kiểm ảnh hưởng ngân sách khi smoke; không gọi nó là parity suy luận GPU.

Các cờ **mới** source/verify, bốn luật nhẹ, và `agent.calc` mặc định tắt. Các cờ
production đã có trong 268 vẫn bật. `scripts/docker_swap.sh` luôn bắt đầu lại từ 268,
không kế thừa cờ thử của ứng viên trước. Khi bật review, script ghi danh sách app known
cụ thể, tránh tìm train trong image. Model tạm `qwen35-4b-soup-98f-7-dpo-a025` chỉ để
kiểm khung; không đại diện quyết định cuối.

| Luật nhẹ (đang TẮT) | Chức năng | Δ FAQ val_app / val_random offline |
|---|---|---|
| `guards.xref_guard` | Chặn ID thấy ở bảng x-ref khác, trả ghi chú nguồn cần đọc | 0,00 / 0,00 |
| `guards.explicit_defaults` | Điền mặc định rõ trong schema cho tool đọc | +0,17 / +0,15 |
| `guards.ask_schema` | Loại tham số hỏi lại không có trong schema; nhắc nếu rỗng | 0,00 / 0,00 |
| `guards.day_filter` | Điền ngày cho tool đọc khi câu nói có đúng một ngày | +0,19 / +0,30 |
| Cả bốn | Phát lại hai lượt 260 | +0,36 / +0,44; tốt/tệ 3/0 app, 2/0 random |

Nguồn: `reports/LIGHT_RULES_1115.md` tại a468590. Phát lại chỉ đo thay call;
phản ứng model với note cần phép đo GPU 267 do quản lý nghiệm thu. LM-99 không chạy GPU
hoặc tự bật luật theo số offline. Cờ source/verify cũng chờ quản lý đọc kết quả 265.

## Chính xác những model vào image

| Đường dẫn trong image | Tham số đếm từ header safetensors |
|---|---:|
| `models/phowhisper-large-base` | 1.543.304.960 |
| `models/phowhisper-large-ft-all-lora` | 57.671.680 |
| `models/phowhisper-large-ft2-all-lora` | 57.671.680 |
| `models/qwen35-4b-soup-98f-7-dpo-a025` (ứng viên tạm) | 4.205.751.296 |
| **Tổng** | **5.864.399.616** |

Dư **135.600.384** dưới trần 6 tỷ. LLM là `qwen3_5_text`, không có tensor vision/LoRA
hoặc adapter rời trong thư mục LLM. Build context tách đúng bốn thư mục, không copy
cả `models/` của máy. `models/app_rank_train.json` là dữ liệu thứ hạng train, không
phải model; Docker đặt `LIFEMATE_APP_RANK=/app/models/app_rank_train.json`.
CPU checker đếm mọi safetensors, kiểm shard/index/offset/kích thước và từ chối checkpoint
lồng, adapter dư hoặc file trọng số không được tính. Header hash không thay thế hash payload.
Biên nhận cục bộ `artifacts/docker/final-cpu.json`; hash toàn trọng số/image do build sau chốt ghi lại.

## Dữ liệu và teacher

Danh sách có số dòng, kích thước, SHA-256 theo từng nhánh nằm trong
`docs/submission_data_candidates.json`. **Tất cả trạng thái pending**, chưa có danh sách
seed/hệ số soup để chọn tập cần gửi. Không đưa nguyên audio, catalog, environment hoặc
JSONL gốc BTC vào archive mã. Các dữ liệu SFT dẫn xuất chỉ đóng gói theo allowlist sau khi chốt.

Teacher khai theo họ: **Claude (Anthropic), GPT (OpenAI), Gemini (Google)**.
D1 hiện có provenance xác nhận Claude viết và Claude giải độc lập để lọc; không mặc định
GPT/Gemini đã sinh D1 hoặc các nhánh khác. Synth v1/v2/v3, rename, calc và guard traces là
biến đổi/sinh bằng mã; D1 diễn đạt lại và phần TTS của D1 phải giữ MANIFEST nguồn riêng.

`src/lifemate/synth/scenarios.py::real_app_ids` chỉ đọc catalog train (sửa từ LM-57,
đã có ở nền 07e01a4). LM-99 tái sinh file 1.618 dòng và phần 2.142 dòng full khớp byte
nguồn cũ; 1.000 synth v3 và metadata cũng khớp byte sau đưa mã vào repo. Điều này chứng minh
bỏ phụ thuộc catalog public không đổi dữ liệu đang dùng; không xoá lịch sử thiết kế đã ghi
trong hồ sơ LM-48/LM-57. Quản lý quyết định cách khai lịch sử đó với BTC.

## Handoff cho người dựng/nộp

Từ checkout này, liên kết/copy đúng các model đã cấp vào `models/`, rồi:

```bash
scripts/docker_swap.sh /duong/dan/QWEN35_MERGED \
  'guards.explicit_defaults=true guards.day_filter=true'
# Chỉ truyền cờ đã được quản lý chốt; không truyền chuỗi cờ => tất cả cờ mới TẮT.
# Khi đã có quyền dựng (không chạy trong LM-99):
TAG=lifemate:final scripts/docker_build.sh
```

Không cần tự đặt lại ASR/ADAPTERS/APP_RANK/LLM: build đọc config và inventory đã kiểm,
kiểm CPU lại trước lệnh Docker. Sửa config tay sau swap sẽ bị từ chối do receipt cũ.
Mỗi lần đổi model/cờ cần chạy swap lại. Những lệnh tiếp theo chỉ cho giai đoạn được cấp quyền:

```bash
IMAGE=lifemate:final scripts/docker_cpu_check.sh
# Smoke GPU và kiểm lặp phải dùng split/private fixture được quản lý cấp, không tự đọc public.
docker run --rm --gpus all --network none \
  -v /duong/dan/private_test:/data/private_test:ro -v /duong/dan/result:/result lifemate:final
```

Cần ghi thêm trước nộp: image digest/tar SHA-256, full payload hashes, model lineage/soup
coefficients, dữ liệu đã chọn, license/revision chính thức, GPU đích/A30, độ trễ/p99/max,
VRAM, lỗi chạy và kết quả kiểm lặp. CPU pass không thay các kiểm chứng này.
