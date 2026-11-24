# LM-57 · Bằng chứng và bàn giao

Ngày 13/11/2026. Nền chính xác: `f1f0220e84a4617ed9a3c5cd90d12cfa232a72f3`.
Nhánh `codex/handin`, checkout riêng dưới `relay-codex/work/handin` vì `.git`
của worktree Relay chỉ được đọc. Không sửa worktree pro-threshold/pro-final.
Ref nền được lấy từ checkout `fable-work/pro-final` có đúng commit yêu cầu;
không xác minh được tip remote qua mạng trong sandbox này.

## Kết quả

- `real_app_ids()` chỉ đọc catalog train. Viewer dữ liệu chỉ còn train;
  test schema `stocks.set_price_alert` dùng schema giả thay đọc public.
- Tích hợp cơ chế tái lập của LM-48 (`2fec441`): thứ tự `set` tất định cho dữ
  liệu mới, replay metadata cho hai file cũ; phục hồi bảng v1 trước `pick_detail`.
  Thêm kiểm tra seed/config/subset và metadata thừa/sai, không đổi mã suy luận.
- Có [REPRODUCE.md](REPRODUCE.md), [allowlist dữ liệu](handin_data.json),
  `scripts/pack_handin.sh` + helper Python và `scripts/soup.py` từ Pro/LM-48.
  Script dùng git archive cho HEAD, teacher `4d7a232` và fold `16e6402`;
  dừng khi checkout chưa commit, thiếu code/file, sai hash, đích đã tồn tại.
  Không quét BTC data; 9 file dẫn xuất, 291.426.305 byte (277,93 MiB),
  mỗi file có hash/số dòng/nguồn trong MANIFEST, kèm SHA256SUMS.
- Không commit dữ liệu BTC, hội thoại sinh ra hay trọng số. Hai file
  `docs/repro/*orders.json` chỉ chứa tên trường/thứ tự/hash, không chứa mẫu.
  Giữ nguyên MANIFEST D1 nguồn; tên teacher trong REPRODUCE chỉ theo họ.

## Bằng chứng dữ liệu không đổi

Tất cả bước dựng lại dùng `LIFEMATE_DATA=/tmp/lm57-train-only`:
chỉ `VERSION` và symlink chỉ đọc tới `train/`, không có mục `public_test`.
Reference nằm trong `fable-eval/pro-threshold/sft/`, chỉ đọc. Generator tạo lại
message từ code và train; không chép message từ reference. Tự chạy lại `find`
cho bộ v1 nhận đúng chuỗi thứ tự metadata đã có của LM-48.

| File đối chiếu | Dòng | Kết quả byte | SHA-256 |
| --- | ---: | --- | --- |
| `train_synth.jsonl` | 1.618 | trùng toàn bộ | `b251018b74a93b1a82019d3fd3d3559898bb54f6736122458b34b4ffb939facb` |
| `train_full_para_v1.jsonl` | 5.133 | phần 2.142 dòng sinh lại + D1 nguồn trùng toàn bộ | `d8363b21c7124f37af26971999947f1be2df9b581567e9240325bd11737031d4` |
| `train_full_para_v1_asrfold.jsonl` | 6.081 | phần trên + 948 dòng fold nguồn trùng toàn bộ | `7a42b39bf9f98ffc1efcd65fa846aa02f4ecfc79772c52973ffd076fa4d03c24` |

[reproduction.json](../reports/lm57/reproduction.json) ghi hash/byte và điều kiện
kiểm. Hai gói 600 synth đều khớp; không có mẫu synth khác cần thay.
D1 và fold ở phép đối chiếu này là **artifact nguồn đã có**, không tuyên bố
đã gọi teacher hay train lại ASR để sinh chúng.

Seed 2026 một mình không đủ tái lập artifact cũ: thứ tự duyệt `set` phụ thuộc
hash seed Python và file nhỏ dùng bảng v1. Chạy generator hiện tại chỉ với
`--synth 600` không phải phép kiểm đúng công thức cũ. Cơ chế replay ghi rõ
giới hạn này; đây không phải chứng minh mọi yếu tố lịch sử đã hợp lệ thể lệ.

## Kiểm tra CPU và audit public

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= LIFEMATE_DATA=/tmp/lm57-train-only \
  PYTHONPATH=src <workspace>/fable-eval/pro-threshold/.venv/bin/python -m pytest -q
<workspace>/fable-eval/pro-threshold/.venv/bin/ruff check .
git diff --check
bash -n scripts/pack_handin.sh
```

**259 passed, 1 skipped** (35,22 giây); ruff sạch. Skip là nhóm Qwen3.5 cần
Transformers 5.x. [pytest](../reports/lm57/pytest.txt), [ruff](../reports/lm57/ruff.txt).
Test pack dùng repo/dữ liệu giả, kiểm archive/hash/số dòng, bỏ qua file BTC nằm
cạnh dữ liệu dẫn xuất, từ chối source bẩn, file thay đổi/thiếu, dữ liệu bị track,
commit thiếu và đích có sẵn; lỗi không để lại gói dở dang.

Đã thử bổ sung nhóm Kaggle/Qwen3.5 bằng môi trường 5.20.2, tokenizer cục bộ và
model tí hon, CPU. Bài smoke 2 rank không mở được TCPStore do sandbox chặn socket
(`PermissionError: Operation not permitted`, `RendezvousConnectionError`).
Dừng lượt thử sau khi xác định trở ngại; **chưa xác nhận DDP/resume nhóm này
trong LM-57**. Mã trainer không đổi. Không dùng GPU, Kaggle, API teacher hay
tải thêm package/model; không chạy đóng gói dữ liệu thật thay Pro.

[public-references.txt](../reports/lm57/public-references.txt) là grep source
Python/shell sau sửa. Các mục còn lại:

- `scripts/asr_latency.py`, `scripts/docker_check.sh`: dự đoán/kiểm suy luận.
- Config public trong `configs/`: cấu hình dự đoán, không input train/synth.
- `src/lifemate/data.py`, `kaggle/run_sft.py`, `app/dataset.py`: chú thích.
- `tests/test_lm25_review.py`: agent giả mang nhãn split public, không đọc BTC.
- `tests/test_normalize.py`: fixture hồi quy đã có gắn ID public; không đọc file.

Các hàm loader chung nhận `split` phục vụ suy luận. Test synth/build_sft chặn
mọi lời gọi `data.*` dùng split khác train. Hai archive lịch sử vẫn giữ generator
cũ theo commit; tài liệu chỉ dùng chúng cho teacher/fold, dùng generator sửa từ
HEAD cho synth. Không tuyên bố grep của HEAD là grep của các archive lịch sử.

## Chưa chắc chắn và việc quản lý cần chốt

1. **Phát hành:** sandbox chặn socket/mạng, chưa push/mở PR. Bàn giao commit,
   patch và bundle chỉ chứa code để Pro push `codex/handin`, PR base
   `max/q35-kaggle-docker`; phải kiểm remote chưa đi tiếp trước khi merge.
2. **Model cuối:** artifact `soup.json` có ứng viên đều 7 seed (0–6), nhưng
   tên image `final-98f` không đủ chứng minh image mang ứng viên đó. Pro chốt
   model, nguồn seed/hệ số, hash image và biên nhận build/check thực tế. LM-57
   không đổi trọng số và không khẳng định tăng điểm.
3. **Nguồn gốc phát triển:** LM-48 đã ghi lịch sử chọn HELDOUT_APPS, chủ đề
   synth và một số luật suy luận có tham chiếu public. Gỡ I/O hiện tại và trùng
   byte chỉ chứng minh dữ liệu có thể dựng từ train mà không cần public;
   quản lý cần quyết định khai báo lịch sử/đối chiếu FAQ với BTC. Giữ luật
   suy luận theo phạm vi đã duyệt, không tự sửa để che nguồn.
4. **Chạy thật:** Pro chạy `scripts/pack_handin.sh DEST` ở checkout sạch,
   kiểm `sha256sum -c SHA256SUMS`, gom image riêng. Khi công thức dữ liệu đổi,
   phải review/cập nhật allowlist/hash trước; script hiện cố định 98f 6.081 dòng.
5. Các thời gian train là log/ước lượng ghi trong REPRODUCE; không có đo GPU
   mới, không xác nhận độ trễ/điểm hay tính tất định Docker trong nhiệm vụ này.
