# Mô tả phương pháp (bản nộp 98)

Bài toán: nghe một yêu cầu tiếng Việt (kèm lịch sử hội thoại có audio), gọi tool trên `ToolEnv` của BTC. Điểm mỗi mẫu là F1 của danh sách call, nhân với việc `state_hash` cuối có khớp hay không. Ràng buộc: dưới 6 tỷ tham số (LLM dưới 5 tỷ), ngữ cảnh 16K, 30 giây mỗi mẫu trên một A30, offline trong Docker.

```
audio ─► ASR ─► agent (LLM, ReAct JSON) ─► middleware và luật ─► ToolEnv ─► tool_calls + state_hash
        PhoWhisper-large           Qwen3-4B-Instruct-2507
        LoRA, gộp fp16             LoRA SFT, gộp bf16
```

| Thành phần | Mô hình | Tham số | Giấy phép gốc |
| --- | --- | --- | --- |
| ASR | `vinai/PhoWhisper-large`, LoRA r=32 gộp vào trọng số | 1.543.304.960 | BSD-3-Clause |
| LLM | `Qwen/Qwen3-4B-Instruct-2507`, LoRA r=16 gộp vào trọng số | 4.022.468.096 | Apache-2.0 |
| Tổng | | **5.565.773.056** | |

Hai mô hình gốc công bố trước 01/10/2026. Lệnh dựng lại từng bước ở [SUBMISSION.md](SUBMISSION.md).

## 1. ASR: PhoWhisper-large fine-tune

- Dữ liệu: lượt cuối và mọi lượt user trong lịch sử của các mẫu `train_rest` (train trừ hai tập kiểm định `val_random` và `val_app`, `src/lifemate/splits.py`): 3.165 câu (4,2 giờ) để học, 200 câu để chọn epoch.
- LoRA r=32 trên mọi lớp chiếu attention và FFN của encoder lẫn decoder, 3 epoch, lr 2e-4, batch 8 × 2, seed 2026 (`scripts/finetune_whisper.py`). Adapter gộp vào checkpoint fp16, nên suy luận không cần peft.
- Nhãn là `audio_text` nguyên dạng (hoa thường, dấu câu, chữ số), để lời ra giống văn bản agent được huấn luyện. PhoWhisper gốc viết thường và đọc số thành chữ.
- Trên 225 mẫu `val_random` (lượt cuối): WER chuẩn hoá số 24,4 → 9,3; tên riêng và số giữ đúng nguyên văn 54,7% → 76,8%.
- Suy luận: greedy, theo lô 16, tiếng Việt, số token tối đa theo độ dài audio để chặn vòng lặp ảo giác.

## 2. Agent: Qwen3-4B SFT, ReAct JSON

**Ngữ cảnh mỗi mẫu** (`src/lifemate/context.py`, `prompts/react_json.md`, `prompts/rules/verified.md`):

- hồ sơ người dùng, `current_time`, và bảng lịch hai tuần (thứ, ngày) để mô hình không phải tự tính "thứ Bảy tuần sau";
- schema nén của các app đã cài: tên tool, loại đọc/ghi, tham số bắt buộc, `x-ref` (ID lấy từ tool nào), trường văn bản tự do;
- lịch sử hội thoại (lời user lấy từ ASR) và sổ ghi các call, kết quả đã có trong lịch sử;
- luật dùng tool rút từ dữ liệu train (khi nào đọc trước khi ghi, khi nào hỏi lại, quy ước tham số call đọc).

**Mỗi bước** LLM trả một JSON `{"thought", "action": "call" | "stop", "name", "arguments"}`. Giải mã bị ép theo JSON Schema: `action` chỉ được là call hoặc stop, `name` chỉ được là một tool đã cài hoặc `sys.ask_missing_info`. Call được chạy thật trên `ToolEnv`, kết quả nối vào hội thoại, tối đa 8 bước. Nhiệt độ 0, seed 0.

**SFT** (`scripts/build_sft.py`, `scripts/train_sft.py`): mỗi mẫu train thành một hội thoại nhiều lượt đúng như lúc chạy: system và user dựng bằng chính agent ở trên, mỗi call gold là một lượt assistant, theo sau là kết quả tool từ `label.tool_results`, kết thúc bằng stop. Mẫu gold rỗng chỉ có một stop. Bản nộp học trên cả train (1.542 mẫu; 3 mẫu bị bỏ vì call gold bị middleware cảnh báo), LoRA r=16, α=32 trên mọi lớp chiếu, 1 epoch, lr 2e-4, cộng dồn 16, độ dài tối đa 12.288 token, seed 0, fp16 trên Kaggle T4. Loss chỉ tính trên lượt assistant.

Trên tập kiểm định (agent học trên `train_rest`, nghe đúng): 34,2 → 61,0 ở `val_random`, 36,1 → 42,9 ở `val_app` (10 app không có trong train, gần với private nhất). Qua ASR thật: 51,8 và 32,5.

## 3. Middleware và luật viết tay (tầng D)

Mọi call của agent đi qua `GuardedEnv` (`src/lifemate/guards.py`, `fixes.py`) trước khi tới `ToolEnv`:

- kiểm tham số theo schema, chuẩn hoá kiểu (số, ngày giờ ISO có múi giờ), bù `end` của lịch từ thời lượng nói ra, cảnh báo ID chưa thấy và call trùng;
- sửa tham số đọc được thẳng từ lời nói mà mô hình 4B hay sai: ngày giờ lệch sáng/chiều hoặc lệch ngày nói ra (`clock_fix`), số tiền VND nói tắt "ba trăm rưỡi" (`money_k`), số phút "từ A tới B" (`span_min`);
- hỏi lại thay vì bịa khi lời không có số tiền hay giờ (`ask_missing`), hoặc khi "hủy bớt một gói" mà có nhiều gói (`pick_one`). Câu hỏi lại được dồn về cuối lượt, để agent vẫn làm xong phần còn lại.

Các luật không đổi call gold nào của 1.545 mẫu train (`tests/test_fixes.py`). Trên toàn bộ val: +2,1 / +0,7 điểm khi nghe đúng, +1,3 / +0,3 qua ASR, không mẫu nào tệ đi.

## 4. Tất định và thời gian

- vLLM 0.33 chạy với `VLLM_BATCH_INVARIANT=1` (kernel bất biến theo batch), giải mã tham lam, seed 0. Không bật thì cùng config chạy lại chỉ giống nhau 12/15 mẫu.
- Agent chạy tuần tự từng mẫu. Giới hạn thời gian mỗi mẫu 25 giây, tính cả ASR: gần hết giờ thì dừng và nộp các call đã gọi. Trên RTX 5060 Ti mẫu chậm nhất khoảng 10 giây, nên giới hạn này chỉ là lưới an toàn và không cắt mẫu nào khi chạy bình thường.
- Docker chạy ba pha: ASR cả split theo lô, rồi khởi động vLLM, rồi agent. Mỗi pha một tiến trình, nên GPU được trả hết trước pha sau.
- Trên RTX 5060 Ti, hai lượt cả public trong Docker tắt mạng ra cùng file từng byte; mỗi mẫu p99 9,0 giây, chậm nhất 10,7; cả lượt 42,4 phút cho 634 mẫu kể cả nạp mô hình.

## 5. Kết quả

| Bản | Mô tả | Public |
| --- | --- | --- |
| `90` | PhoWhisper-large-ft rồi Qwen3-4B SFT trên `train_rest` | 42,15 |
| `91` | như 90, SFT trên cả train | 44,70 |
| `98` | như 91, thêm luật viết tay (khác 91 ở 11/634 mẫu); image Docker ra đúng file này | chưa có điểm khi viết |

Chi tiết từng thí nghiệm ở [BASELINE.md](BASELINE.md); kiến trúc và lý do chọn ở [PROPOSAL.md](PROPOSAL.md).
