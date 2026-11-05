# Baseline mô-đun: đổi phương án từng tầng, so sánh trên cùng một thước đo

Khung này để thử nhiều kiến trúc mà không sửa code: mỗi tầng của pipeline có nhiều phương án, chọn bằng một dòng trong file config. Mọi lượt chạy đi qua cùng bộ chấm và cùng tập kiểm định, rồi được đặt cạnh nhau trong web-app.

## 1. Các tầng và nơi đổi phương án

| Tầng | Khoá config | Phương án hiện có | Code |
| --- | --- | --- | --- |
| A. ASR | `asr.name`, `asr.model` | `gold`, `hf_whisper` (Whisper-turbo, PhoWhisper, PhoASR-whisper-small), `faster_whisper`, `chunkformer`, `dialect_phonetic` (chờ mã); `model` nhận cả checkpoint fine-tune trong `models/` | `src/lifemate/asr/adapters.py`, `scripts/finetune_*.py` |
| A'. Hậu xử lý | `asr_post` | `strip_fillers`, `phonetic_entity_fix` | `src/lifemate/asr/post.py` |
| B. Dựng ngữ cảnh | `agent.timetable`, `agent.schema`, `agent.history`, `agent.ledger` | bảng lịch 3 ngày hoặc 2 tuần; schema thô hoặc nén; sổ ghi lịch sử bật/tắt | `src/lifemate/context.py` |
| C. Agent | `agent.name`, `agent.llm`, `agent.prompt`, `agent.rules` | `empty`, `oracle`, `react_json`, `native_tools`; LLM bất kỳ qua API chuẩn OpenAI | `src/lifemate/agent/` và `prompts/` |
| D. Middleware | `guards.*` | validate, chuẩn hoá, bù `end` lịch, chặn ID (off, warn, block), chặn gọi trùng; luật sửa tham số theo lời nói (`clock_fix`, `money_k`, `span_min`, `ask_missing`, `pick_one`) | `src/lifemate/guards.py`, `fixes.py` |
| Đo | `subset`, `limit`, `repeats` | `val_random`, `val_app` (10 app giữ lại), `train_rest`, `all`; lặp để đo độ ổn định | `src/lifemate/splits.py`, `scoring.py` |

`lifemate list` in ra mọi phương án đã đăng ký và trạng thái của chúng.

## 2. Cài đặt

```bash
uv sync --extra asr --extra app --extra dev --extra chunkformer --extra sft
```

```bash
uv venv --python 3.12 .venv-vllm && uv pip install --python .venv-vllm/bin/python vllm
```

```bash
.venv/bin/python scripts/download_models.py
```

Dữ liệu BTC đặt ở `zalo-ai-2026/data/lifemate/` (tìm tự động ở các thư mục cha), hoặc trỏ biến `LIFEMATE_DATA` tới đó.

GPU RTX 50xx cần torch bản CUDA 12.8 trở lên; `pyproject.toml` đã ghim index `cu128` cho torch và torchaudio.

## 3. Chạy

```bash
.venv/bin/lifemate run configs/experiments/01_oracle.yaml
```

```bash
.venv/bin/lifemate run configs/experiments/10_qwen3-4b_gold_react.yaml --set limit=50 --set agent.max_steps=6
```

```bash
scripts/serve_vllm.sh
```

```bash
scripts/run_queue.sh configs/experiments/1*.yaml configs/experiments/3*.yaml
```

```bash
.venv/bin/streamlit run app/compare.py
```

```bash
.venv/bin/pytest -q
```

- Mỗi lượt chạy ghi vào `runs/<thời gian>_<tên config>/`: `config.yaml`, `predictions.jsonl` (đúng định dạng nộp), `transcripts.jsonl`, `samples.jsonl` (điểm, nhãn lỗi, vết agent từng mẫu), `metrics.json`.
- Transcript được lưu theo khoá config ASR trong `cache/asr/`, nên đổi agent hay middleware không phải chạy lại ASR.
- `scripts/run_queue.sh` chạy lần lượt nhiều config và bỏ qua config đã có kết quả; bị ngắt thì chạy lại đúng lệnh đó để nối tiếp.
- Sửa cách tính độ đo ASR thì chạy `lifemate rescore runs/<run> ...` để tính lại từ `transcripts.jsonl`, không phải chạy lại mô hình.
- vLLM và ASR cùng dùng GPU 16 GB: chạy hết các config ASR trước, hoặc tắt server vLLM khi cần GPU cho ASR.
- `lifemate validate runs/<run>` kiểm schema file nộp bằng `zaic_tools` của đúng split.

## 4. Thêm một phương án mới

Ví dụ thêm một ASR mới, ba bước.

1. Viết lớp trong `src/lifemate/asr/adapters.py`:

   ```python
   @register("asr", "my_asr", status="needs_deps")
   class MyASR(ASR):
       """Một dòng mô tả: hiện trong `lifemate list` và tab Checklist."""

       def __init__(self, model="org/checkpoint", beam=4, **_):
           ...  # nạp mô hình

       def transcribe(self, reqs: list[Request]) -> list[Result]:
           return [Result(text=..., alts=[...]) for r in reqs]  # r.path là file wav 16 kHz, r.hint là chuỗi mồi
   ```

2. Tạo `configs/experiments/27_asr_my-asr.yaml`:

   ```yaml
   # Một dòng nói thí nghiệm này trả lời câu hỏi gì (hiện ở tab Checklist).
   extends: ../base.yaml
   asr: {name: my_asr, model: org/checkpoint, beam: 4}
   agent: {name: none}
   ```

3. Chạy `lifemate run` rồi mở web-app: tab ASR đặt WER, CER, độ giữ thực thể cạnh các ASR khác.

Agent, hậu xử lý và middleware làm giống vậy với `@register("agent", ...)`, `@register("asr_post", ...)` hoặc một khoá mới trong `guards`. Đổi luật trong prompt không cần code: sửa hoặc thêm file trong `prompts/rules/` rồi trỏ `agent.rules` tới nó.

## 5. Quy tắc so sánh

- Chỉ so điểm giữa các lượt chạy cùng `subset`. `val_app` là con số gần với private nhất; `val_random` đo việc học quy ước.
- Đổi một khoá mỗi lần so với một lượt chạy gốc. Tab Kiến trúc tô các dòng config khác nhau để thấy ngay mình đang so cái gì.
- Đo độ ổn định bằng `repeats: 3`: tỉ lệ mẫu ra cùng call và cùng hash giữa các lần. Cần 100% trước khi nộp Docker.
- Đo thời gian bằng `workers: 1`. Với `workers` lớn hơn, thời gian mỗi mẫu bị lẫn do chạy song song.
- Oracle (`01`) phải luôn ra 100. Một bảo vệ mới làm oracle tụt điểm là bảo vệ đang chặn nhầm call đúng.
- Hai mô hình SFT cùng công thức, chỉ khác seed, lệch nhau tới 3 điểm trung bình hai tập (độ lệch chuẩn 1,2 trên cả tập, 2,2 trên 100 mẫu; mục 7). Một lượt huấn luyện thắng dưới 3 điểm trên 100 mẫu chưa đủ để đổi công thức: chạy cả tập, hoặc huấn luyện lại vài seed.

## 6. Kết quả vòng đầu (03/11/2026)

Máy: RTX 5060 Ti 16 GB (không phải A30). Agent: Qwen3-4B-Instruct-2507 chỉ dùng prompt, qua vLLM 0.33. Các lượt agent dùng cùng 100 mẫu cố định của `val_random` (trừ dòng `val_app`); ASR đo trên 225 mẫu `val_random`. Với 100 mẫu, chênh lệch dưới khoảng 4 điểm chưa đáng tin.

**Khung đo (1.545 mẫu train)**

| Config | Điểm |
| --- | --- |
| `00` không gọi gì | 7,18 |
| `01` oracle qua middleware mặc định | 100,00 |
| `02` oracle qua middleware của HTML (chặn ID chỉ theo lượt hiện tại) | 95,45 |

**ASR (225 mẫu, lượt cuối)**

| Config | WER chuẩn hoá số | Giữ thực thể đúng cả dấu |
| --- | --- | --- |
| `23` ChunkFormer (110 triệu, CC BY-NC) | **20,5** | **69,5** |
| `21` PhoWhisper-large (1,55 tỷ) | 24,4 | 54,7 |
| `22` PhoASR-whisper-small (0,24 tỷ) | 26,9 | 52,6 |
| `24` Whisper-turbo qua CTranslate2 | 30,5 | 45,3 |
| `20` Whisper-turbo qua transformers | 31,1 | 45,3 |
| `25` Whisper-turbo + mồi tên + sửa âm vị | 33,2 | 47,4 |
| `27` Whisper-turbo + chỉ sửa âm vị | 32,0 | 43,2 |
| `28` ChunkFormer + sửa âm vị | 21,5 | 66,3 |

Trần của cột thực thể khoảng 95%, vì gold đôi khi viết dấu kiểu khác ("hoà" và "hòa"). Whisper-turbo gốc ảo giác trên một phần mẫu ("Hãy subscribe cho kênh…"). Bước sửa âm vị bằng luật tay làm giảm độ giữ thực thể, nên chưa nên bật.

**ASR fine-tune trên audio train (225 mẫu `val_random`, lượt cuối)**

| Config | WER chuẩn hoá số | Giữ thực thể đúng cả dấu | Giây ASR mỗi mẫu | Mốc gốc |
| --- | --- | --- | --- | --- |
| `29_asr_chunkformer-ft` | **9,3** | **77,9** | 0,42 (CPU) | `23`: 20,5 / 69,5 / 0,41 (CPU) |
| `29_asr_phowhisper-large-ft` | **9,3** | 76,8 | 0,54 (GPU) | `21`: 24,4 / 54,7 / 0,57 (GPU) |

- Dữ liệu: lượt cuối và mọi lượt user trong lịch sử của các mẫu `train_rest`, 3.165 câu (4,2 giờ) để học và 200 câu để chọn epoch (`src/lifemate/asr/train_data.py`). Mẫu `val_random` và `val_app` không vào huấn luyện (có test), nên bảng này so thẳng được với bảng trên. Chỉ 1/225 câu `val_random` trùng nguyên văn một câu trong train.
- ChunkFormer: học toàn bộ 110 triệu tham số, CTC với chunk động như bản gốc, 10 epoch, lr 3e-5, SpecAugment; khoảng 7 phút trên RTX 5060 Ti. Nhãn viết thường, số đọc thành chữ, vì từ vựng của mô hình không có chữ số và chữ hoa. WER dev giảm từ 20,6 xuống 9,0. Trên `val_random`, 184/225 câu tốt hơn và 8 câu kém hơn; các lượt lịch sử giảm từ 20,7 xuống 8,9.
- PhoWhisper-large: LoRA r=32 trên mọi lớp chiếu của encoder và decoder (58 triệu tham số học), 3 epoch, lr 2e-4, khoảng 75 phút; adapter đã gộp vào checkpoint fp16 nên suy luận không cần peft. Nhãn là `audio_text` nguyên dạng, nên lời ra có hoa thường và dấu câu như transcript gold (PhoWhisper gốc viết thường). WER dev giảm từ 27,0 xuống 8,7 và vẫn đang giảm ở epoch cuối. Trên `val_random`, 191/225 câu tốt hơn và 8 câu kém hơn; các lượt lịch sử giảm từ 27,2 xuống 9,1. Không thấy câu ảo giác lặp.
- Hai bản fine-tune ngang nhau về WER (bản này hơn ở 75 câu, kém ở 66 câu). PhoWhisper-large-ft theo BSD-3 nên nộp được, và 1,55 tỷ cộng Qwen3-4B vẫn dưới 6 tỷ tham số. ChunkFormer-ft nhỏ, chạy được trên CPU, nhưng vướng giấy phép.
- Giấy phép: checkpoint fine-tune từ ChunkFormer vẫn theo CC BY-NC 4.0 của mô hình gốc. Phải được BTC đồng ý trước khi đưa vào bài nộp.
- Adapter `chunkformer` hiện chạy trên CPU (không chuyển mô hình lên GPU), nên cột giây của `23` và `29_asr_chunkformer-ft` là thời gian CPU (16 nhân). Hai dòng PhoWhisper đo trên GPU.
- Checkpoint nằm ở `models/` (không commit). Tạo lại bằng:

  ```bash
  uv sync --extra asr --extra chunkformer --extra train
  ```

  ```bash
  .venv/bin/python scripts/finetune_chunkformer.py --out models/chunkformer-ctc-large-vie-ft
  ```

  ```bash
  .venv/bin/python scripts/finetune_whisper.py --model vinai/PhoWhisper-large --out models/phowhisper-large-ft
  ```

**Agent (100 mẫu)**

| Config | Điểm | Đúng hash | Ghi chú |
| --- | --- | --- | --- |
| `10` gold text, ReAct JSON, giải mã theo schema | 34,2 | 54% | mốc gốc |
| `16` như 10, chặn hẳn ID chưa thấy | **37,1** | 56% | +2,9 |
| `17` như 10, chỉ ép JSON (không ép tên tool) | 33,9 | 55% | |
| `13` như 10, luật tối thiểu | 31,7 | 50% | −2,5: luật khai thác từ dữ liệu có ích |
| `14` như 10, ngữ cảnh kiểu HTML | 32,5 | 54% | p95 83,6 giây (8 luồng) vì schema thô dài |
| `12` tool calling gốc của Qwen | 32,5 | 46% | |
| `11` gold text trên `val_app` | 36,1 | 48% | mô hình chưa học app nào nên chưa có khoảng cách |
| `31` PhoWhisper-large → agent | 23,9 | 39% | ASR làm mất khoảng 10 điểm |
| `33` ChunkFormer → agent | 21,3 | 42% | |
| `30` Whisper-turbo → agent | 20,8 | 42% | |
| `32` PhoASR-whisper-small → agent | 19,6 | 35% | |

Lỗi lớn nhất của `10` là sai tham số call đọc (20/100 mẫu, ví dụ truyền `location` cho thời tiết ở cả 8/8 lần dù luật cấm), rồi chọn làm thay vì hỏi lại (9) và tự bịa ID (9). Đây là các quy ước phải học bằng SFT, prompt không ép được.

**Thời gian và độ ổn định (30 mẫu, gold text)**

| Config | Trung vị | p95 | Lâu nhất | Ổn định qua 3 lần |
| --- | --- | --- | --- | --- |
| `18` tuần tự | 3,9 giây | 8,5 giây | 12,6 giây | |
| `18` tuần tự, `DETERMINISTIC=1` | 4,4 giây | 10,6 giây | 14,9 giây | |
| `15` 8 luồng | | | | 66,7% |
| `19` 8 luồng, `DETERMINISTIC=1` | | | | **100%** |

Mặc định vLLM không tất định kể cả khi `temperature: 0` (chạy tuần tự cũng chỉ 12/15 mẫu giống nhau). Bài nộp Docker phải chạy server với `DETERMINISTIC=1`.

## 7. SFT LoRA cho agent (nấc 3, 04/11/2026)

```bash
.venv/bin/python scripts/build_sft.py
```

```bash
.venv/bin/python scripts/build_sft.py --rename 0.5 --out sft/train_rename.jsonl
```

```bash
.venv/bin/python scripts/train_sft.py --data sft/train.jsonl --out models/qwen3-4b-sft
```

```bash
DETERMINISTIC=1 MODEL=models/qwen3-4b-sft scripts/serve_vllm.sh
```

```bash
SET="limit=100" scripts/run_queue.sh configs/experiments/40_*.yaml configs/experiments/41_*.yaml
```

- Dữ liệu lấy từ quỹ đạo gold của 1.018 mẫu `train_rest`, không đụng `val_random` và `val_app`. Mỗi mẫu là một hội thoại nhiều lượt, đúng từng chữ các tin nhắn ReactJSON gửi lúc chạy (`tests/test_sft.py` kiểm điều này). Mỗi call gold là một lượt assistant `{"thought": "", "action": "call", ...}`, kết quả lấy từ `label.tool_results`, cuối cùng là một `stop`.
- Tổng cộng 3.328 lượt assistant, 89 mẫu gold rỗng, 159 mẫu có `sys.ask_missing_info`. Bỏ `train_1696` vì gold bịa `chat_id` nên middleware cảnh báo, và quan sát lúc chạy sẽ khác lúc học.
- `--rename 0.5` đổi tên một nửa số mẫu (proposal 5.5): app thành tên kiểu thương hiệu, từng mảnh tên tool và tham số thành từ đồng nghĩa (giữ `_id`). Việc đổi tên nhất quán trong schema, luật, lịch sử và đáp án; `sys` giữ nguyên.
- LoRA r=16, alpha=32 trên mọi lớp tuyến tính (33 triệu tham số), 1 epoch, lr 2e-4, cộng dồn 16 mẫu (64 bước). Mỗi mẫu dài p50 4.130 token; cả epoch 4,3 triệu token, nhưng chỉ 102 nghìn token có loss (2,4%), phần còn lại là system prompt. Tốc độ 600 token/s, đỉnh 11,2 GB, 2 giờ mỗi mô hình. Loss đi từ 1,09 xuống 0,045 (đổi tên: 0,041).
- `models/` và `sft/` không vào git. `10` và `11` chạy trên server chưa tất định; `40` đến `43` chạy với `DETERMINISTIC=1`.

**Kết quả (cùng 100 mẫu mỗi tập như `10` và `11`)**

| Config | Mô hình | Tập | Điểm | Đúng hash | Mẫu đúng hẳn | Dừng ngay dù gold có call | p95 (8 luồng) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `10` | gốc | `val_random` | 34,2 | 54% | 21 | 2 | 33,5 giây |
| `40` | SFT | `val_random` | **61,0** | **73%** | **55** | 3 | 29,2 giây |
| `42` | SFT đổi tên | `val_random` | 55,8 | 67% | 50 | 6 | 29,3 giây |
| `11` | gốc | `val_app` | 36,1 | 48% | 27 | 5 | 32,2 giây |
| `41` | SFT | `val_app` | **42,9** | **54%** | **36** | 12 | 25,9 giây |
| `43` | SFT đổi tên | `val_app` | 40,3 | 51% | 34 | 10 | 21,4 giây |

Chênh lệch ghép cặp trên cùng mẫu (± là sai số chuẩn):

| So sánh | `val_random` | `val_app` | Khoảng cách `val_random` − `val_app` |
| --- | --- | --- | --- |
| Gốc (`10`, `11`) | | | −1,9 |
| SFT so với gốc | +26,8 ± 5,0 | +6,8 ± 4,3 | +18,1 (tăng 19,9 ± 6,6) |
| SFT đổi tên so với SFT | −5,2 ± 3,1 | −2,6 ± 2,9 | +15,5 (giảm 2,5 ± 4,2, trong nhiễu) |

- **Sau SFT, khoảng cách tăng mạnh**, khoảng 3 lần sai số chuẩn. SFT học được quy ước của các app đã thấy: sai tham số đọc giảm từ 20 xuống 8, cảnh báo middleware từ 18 xuống 2. Nhưng phần học được gần như không chuyển sang app mới.
- **Đổi tên một nửa số mẫu không thu hẹp khoảng cách**, và kéo `val_random` xuống khoảng 5 điểm.
- **Nguyên nhân chính là cách chia tập.** 543/1.019 mẫu `train_rest` có cài ít nhất một app giữ lại (riêng `medication` có 234 mẫu), nhưng gold không bao giờ gọi các app đó, vì mẫu nào gọi thì đã thuộc `val_app`. SFT thấy `loyalty.redeem`, `learning.log_progress`... trong schema hàng trăm lần, chưa lần nào được gọi, nên học cách bỏ qua chúng.
- Dấu hiệu rõ nhất là số mẫu `val_app` mà mô hình `stop` ngay bước đầu dù gold có call: từ 5 tăng lên 12. Ví dụ "Cho anh vô cuộc họp Roadmap Review sáng nay luôn đi em" khi `video_meeting` đã cài sẵn.
- Đổi tên không chữa được việc này: nửa số mẫu còn lại vẫn giữ tên thật, và đổi chữ không đổi việc tool đó chưa bao giờ được gọi. Vì vậy `val_app` lúc này đo "app mô hình được dạy là bỏ qua", khắt khe hơn "app hoàn toàn mới".
- **Thí nghiệm tiếp theo:** gỡ các app giữ lại khỏi `installed_apps` và schema của mẫu SFT, rồi đo lại `41` và `43`. Khi đó `val_app` mới thực sự đo app chưa thấy.
- **Thời gian:** sau SFT, `thought` để trống và số lượt gọi LLM giảm từ 3,26 xuống 3,0. p50 đi từ 10,9 xuống 9,6 giây, p95 dưới 30 giây ngay cả khi chạy 8 luồng.

**Huấn luyện song song trên Kaggle, đánh giá ở máy nhà**

Huấn luyện chiếm 2 giờ mỗi mô hình, đánh giá chỉ 6 phút, nên các biến thể SFT được huấn luyện trên Kaggle (T4 x2, mỗi GPU một thí nghiệm), chỉ tải adapter LoRA về.

```bash
.venv/bin/python scripts/build_sft.py --drop-heldout --out sft/train_drop.jsonl
```

```bash
scripts/kaggle_pack.sh
```

```bash
.venv/bin/python scripts/merge_lora.py models/qwen3-4b-sft-drop-lora models/qwen3-4b-sft-drop
```

```bash
SET="limit=100" scripts/gpu_job.sh eval models/qwen3-4b-sft-drop configs/experiments/44_*.yaml configs/experiments/45_*.yaml
```

- `--drop-heldout` gỡ app giữ lại khỏi `installed_apps` và bỏ 80 mẫu có lịch sử gọi chúng: còn 938 mẫu, 3,84 triệu token.
- `sft/kaggle/` (do `kaggle_pack.sh` dựng) tải lên làm dataset Kaggle **riêng tư**; `kaggle/run_sft.py` liệt kê các thí nghiệm (`drop`, `drop-rename`, `drop-rename-all`, `drop-half`, `t4-check`) và configs 44 đến 53 tương ứng.
- T4 không có bf16 thật nên `train_sft.py` tự chuyển sang fp16 có GradScaler; đã thử 2 bước trên máy nhà: loss hữu hạn, đỉnh 9,8 GB. `t4-check` lặp lại 40/41 để biết kết quả Kaggle có so được với máy nhà không.
- Đo trên Kaggle T4: 280 đến 320 token/s mỗi GPU (khoảng một nửa máy nhà), đỉnh 10,6 GB. `full` (97 bước) mất khoảng 6 giờ, một lượt `drop` khoảng 3,5 giờ. Ảnh Kaggle có `torchao` 0.10 mà peft 0.21 từ chối, nên notebook phải gỡ nó trước (xem docstring của `kaggle/run_sft.py`).
- Gộp adapter chạy trên CPU (khoảng 20 giây), không cần khoá GPU.
- Bản nộp: `--subset all` dựng dữ liệu từ cả 1.545 mẫu (`sft/train_full.jsonl`, 1.542 mẫu sau khi bỏ 3 mẫu bị middleware cảnh báo); thí nghiệm `full` và `full-rename-all` của `run_sft.py`, nộp bằng configs 91 và 92. Các mô hình này đã học cả `val_random` và `val_app`, nên chỉ đo được trên bảng xếp hạng public.
- Đo lại trên Kaggle (05/11): một đợt hai thí nghiệm `train.jsonl` mất 3 giờ 57 đến 4 giờ 12, đợt `full` và `full-rename-all` 6 giờ 40. Tải adapter bằng Kaggle CLI (`kaggle kernels output <chủ>/<notebook>/<phiên bản> -p <thư mục> --file-pattern '^[^/]+\.(zip|log)$'`) với token của một tài khoản có quyền xem notebook.

**Gỡ app giữ lại, đổi tên mọi mẫu (configs 44, 45, 48, 49): không dùng**

| 100 mẫu, gold text | `val_random` | `val_app` |
| --- | --- | --- |
| SFT (`40`, `41`) | 61,0 | 42,9 |
| drop (`44`, `45`) | 56,1 (−4,9 ± 3,8) | 42,5 (−0,4 ± 3,0) |
| drop-rename-all (`48`, `49`), so với drop | 52,4 (−3,7 ± 3,0) | 41,5 (−1,0 ± 2,7) |

- Gỡ app giữ lại khỏi schema không làm `val_app` tăng, và đổi tên mọi mẫu cũng không. Mô hình chưa đọc schema của app chưa thấy tốt hơn khi bớt được thói quen bỏ qua chúng.
- Các hiệu này đều trong dao động theo seed (dưới đây), nên chỉ là bằng chứng yếu, nhưng không có tín hiệu nào đủ để đổi: bản nộp v2 dùng `full`, không đổi tên. `drop-half` và `drop-rename` (46, 47, 50, 51) chưa chạy xong vì tài khoản Kaggle thứ 3 không đăng nhập lại được.

**Dao động theo seed và fp16 trên T4 (configs 52, 53, 80-87)**

Cùng công thức của `40` (dữ liệu `train.jsonl`, 1 epoch), huấn luyện lại trên Kaggle T4 bằng fp16 với seed 0 (`t4-check`) và seed 1-3 (`train_sft.py --seed N`), rồi chấm trên **cả** `val_random` (225) và `val_app` (301). `86`, `87` là `40`, `41` (v1: seed 0, bf16 ở máy nhà) trên cả tập, để có mốc.

| Mô hình | seed | Độ chính xác | `val_random` | `val_app` | Trung bình | Trung bình trên 100 mẫu |
| --- | --- | --- | --- | --- | --- | --- |
| v1 (`86`, `87`) | 0 | bf16, máy nhà | **59,2** | 47,6 | 53,4 | 52,0 |
| `t4-check` (`52`, `53`) | 0 | fp16, T4 | 58,5 | **48,6** | **53,6** | 50,0 |
| `seed1` (`80`, `81`) | 1 | fp16, T4 | 55,7 | 45,6 | 50,6 | 52,0 |
| `seed2` (`82`, `83`) | 2 | fp16, T4 | 55,5 | 47,7 | 51,6 | 48,7 |
| `seed3` (`84`, `85`) | 3 | fp16, T4 | 57,2 | 46,9 | 52,0 | 47,0 |
| độ lệch chuẩn | | | 1,6 | 1,1 | **1,2** | 2,2 |

- **fp16 trên T4 không kém bf16.** Cùng seed 0, hiệu ghép cặp trên cả tập là −0,7 ± 2,0 (`val_random`) và +1,0 ± 1,6 (`val_app`). Đường loss gần như trùng: bước 1 là 1,0878 và 1,0883, cuối 0,0438 và 0,0447. Mô hình huấn luyện trên Kaggle dùng được như mô hình máy nhà.
- **Chỉ đổi seed, trung bình hai tập lệch tới 3 điểm** (50,6 đến 53,6). Trên 100 mẫu, độ lệch chuẩn gần gấp đôi, và thứ hạng còn đảo: `seed1` ngang v1 trên 100 mẫu (52,0) nhưng kém 2,8 điểm trên cả tập. Seed 0 có trung bình cao nhất ở cả hai độ chính xác, có thể chỉ do may; v1 và cả hai bản nộp v2 đều dùng seed 0.
- Vì vậy các quyết định trước đây dựa trên một lượt huấn luyện và 100 mẫu (đổi tên −5,2, drop −4,9, rename-all −3,7, asrmix −1,4 đầu-cuối) chỉ là bằng chứng yếu: hướng đều là âm nên không đổi kết luận, nhưng hiệu nhỏ hơn khoảng 3 điểm không tách được khỏi seed. Quy tắc ở mục 5.

## 8. Ghép đầu-cuối khi cả hai tầng đã tinh chỉnh (04/11/2026)

ASR fine-tune bằng `scripts/finetune_whisper.py` và `scripts/finetune_chunkformer.py` (`models/phowhisper-large-ft`, `models/chunkformer-ctc-large-vie-ft`), agent là `models/qwen3-4b-sft` của `40`. PhoWhisper và vLLM không cùng vừa 16 GB, nên ASR chạy trước một mình để điền cache (configs `29_*_val-app`), rồi mới bật vLLM.

```bash
SET="limit=100" scripts/gpu_job.sh run configs/experiments/29_asr_phowhisper-large-ft_val-app.yaml configs/experiments/29_asr_chunkformer-ft_val-app.yaml
```

```bash
SET="limit=100" scripts/gpu_job.sh eval models/qwen3-4b-sft configs/experiments/5[4-7]_*.yaml
```

| Config | ASR | Agent | Tập | Điểm | Đúng hash | WER chuẩn hoá số | Mất do ASR (ghép cặp) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `10` | chữ đúng | gốc | `val_random` | 34,2 | 54% | | |
| `31` | PhoWhisper-large gốc | gốc | `val_random` | 23,9 | 39% | 24,5 | −10,3 ± 4,4 |
| `33` | ChunkFormer gốc | gốc | `val_random` | 21,3 | 42% | 21,0 | −12,9 ± 3,7 |
| `40` | chữ đúng | SFT | `val_random` | 61,0 | 73% | | |
| `54` | PhoWhisper-large-ft | SFT | `val_random` | **51,8** | **65%** | 10,3 | **−9,2 ± 3,4** |
| `56` | ChunkFormer-ft | SFT | `val_random` | 41,7 | 60% | 10,4 | −19,3 ± 4,0 |
| `41` | chữ đúng | SFT | `val_app` | 42,9 | 54% | | |
| `55` | PhoWhisper-large-ft | SFT | `val_app` | **32,5** | 39% | 11,6 | **−10,4 ± 3,4** |
| `57` | ChunkFormer-ft | SFT | `val_app` | 28,9 | 38% | 11,3 | −14,0 ± 3,3 |

- **Pipeline thật tăng từ 23,9 lên 51,8** trên `val_random` (+27,9 ± 5,4 ghép cặp, `31` so với `54`). `54`/`55` là tổ hợp đang tốt nhất và là bản nộp đầu (`90`): 634 dòng, `zaic_tools --validate` không báo lỗi; 73 mẫu không gọi tool, 41 mẫu hỏi lại; p50 9,0 giây, p95 27,8 giây khi chạy 8 luồng, lâu nhất 55 giây. **Điểm public: 42,15**, gần đúng bằng trung bình hai tập kiểm định ((51,8 + 32,5) / 2 = 42,2), nên trung bình `val_random` và `val_app` là ước lượng điểm public dùng được, và là cột xếp hạng của tab "Kiến trúc tốt nhất".
- **ASR vẫn làm mất khoảng 10 điểm**, dù WER giảm từ 24,5 xuống 10,3. Tính theo tỉ lệ thì phần mất giảm một nửa (30% điểm của `10`, giờ 15% điểm của `40`): agent giỏi hơn nên mỗi lỗi nghe tốn nhiều điểm hơn. Lỗi thêm vào chủ yếu là đứng yên hoặc thiếu lệnh ghi (`54` so với `40`: thiếu lệnh ghi +4, làm → đứng yên +3, hỏi lại → đứng yên +2), hiếm khi làm sai.
- **ChunkFormer-ft ngang PhoWhisper-ft khi đo ASR một mình** (WER 9,27 và 9,31, giữ thực thể 77,9% và 76,8% trên 225 mẫu), **nhưng khi ghép thì kém hơn 10,1 ± 4,3 điểm** trên `val_random` (3,6 ± 3,6 trên `val_app`). Nó ra chữ thường, không dấu câu, số đọc thành chữ ("hai mươi sáu độ" thay cho "26 độ"), khác kiểu văn bản gold mà agent SFT đã học; PhoWhisper-ft ra đúng kiểu gold. Có 13 mẫu PhoWhisper-ft đúng hẳn mà ChunkFormer-ft sai, 5 mẫu ngược lại. Kèm giấy phép CC BY-NC, ChunkFormer bị loại.

**Cách chọn tổ hợp: tách theo đồ thị phụ thuộc, không thử mọi tổ hợp**

- Các tầng nối thành chuỗi A (ASR) → hậu xử lý → C (LLM, prompt, schema) → D (guards). Chọn theo đúng thứ tự đó: mỗi tầng xếp hạng bằng độ đo rẻ của riêng nó, với tầng trên cố định ở phương án tốt nhất. ASR đo bằng WER và giữ thực thể, không cần LLM; agent đo trên chữ đúng (`asr: gold`).
- Sau đó chỉ chạy ghép vài phương án đầu của mỗi cặp tầng kề nhau. Số lượt cần chạy là tổng chứ không phải tích: 8 ASR × 10 agent × 4 guards là 320 tổ hợp, còn cách này cần khoảng 25 lượt.
- Giả định tách tầng được kiểm bằng cột "mất do ASR". Với PhoWhisper, phần mất gần như không đổi giữa agent gốc và agent SFT (−10,3 và −9,2), nên chọn riêng từng tầng là đủ. Với ChunkFormer, phần mất tăng từ −12,9 lên −19,3, tức là có tương tác giữa hai tầng. Vì vậy cặp ASR × agent luôn phải chạy ghép, và điểm ghép mới là thứ quyết định.
- Guards phụ thuộc LLM: `16` (chặn hẳn ID chưa thấy) thêm 2,9 điểm cho mô hình gốc, nhưng với agent SFT (`58`, `59`) không đổi một mẫu nào trong 200 (61,0 và 42,9 như `40`, `41`): SFT không còn bịa ID nên guard thừa. Giữ mặc định.

**SFT chịu lỗi nghe (configs 60-63): không dùng**

- ASR fine-tune đã học chính audio `train_rest`: PhoWhisper-large-ft chép `train_rest` với WER 2,0 (val_random 9,3), nên bản chép đó gần như sạch, không dạy được agent xử lý khi nghe sai.
- `hf_whisper_noisy` (`src/lifemate/asr/augment.py`) thêm nhiễu trắng vào audio trước khi nhận dạng, chỉ để dựng dữ liệu. Thử trên 100 mẫu `train_rest`: WER 4,7 ở 10 dB, 8,4 ở 5 dB, 11,8 ở 2 dB; chọn 2 dB, gần mức audio chưa nghe mà vẫn giữ kiểu chữ của mô hình (chữ hoa, dấu câu, chữ số), khác với ASR gốc ra chữ thường. Trên cả 1.019 mẫu, 2 dB ra WER 14,3, nặng hơn ASR thật trên val (9,3 val_random, 12,0 val_app).
- `build_sft.py --asr-config configs/experiments/29_asr_phowhisper-large-ft-noisy_train-rest.yaml --asr-frac 0.5` thay chữ đúng bằng bản chép có nhiễu ở 532/1.018 mẫu, đáp án vẫn là call gold. Mô hình `qwen3-4b-sft-asrmix`, cùng công thức với `40`.

Kết quả trên 100 mẫu đầu mỗi tập, hiệu ghép cặp theo mẫu:

| | gold, val_random | gold, val_app | ASR, val_random | ASR, val_app | trung bình đầu-cuối |
| --- | --- | --- | --- | --- | --- |
| SFT (`40`, `41`, `54`, `55`) | 61,0 | 42,9 | 51,8 | 32,5 | 42,2 |
| SFT trộn ASR (`60`-`63`) | 51,9 | 40,0 | 48,2 | 33,4 | 40,8 |
| hiệu | −9,1 ± 3,1 | −2,9 ± 3,0 | −3,6 ± 3,3 | +0,8 ± 3,0 | −1,4 |

- Phần mất do ASR thu hẹp (val_random 9,2 → 3,7 điểm, val_app 10,4 → 6,7) chủ yếu vì nghe đúng kém đi, không phải vì nghe sai được xử lý tốt hơn. Điểm đầu-cuối không tăng, nên bản nộp giữ `qwen3-4b-sft`.
- Có thay đổi đúng hướng: trên val_app gold, 4 mẫu chuyển từ đứng yên sang làm. Nhưng sai ID và sai tham số ghi tăng (+3, +4 trên val_random gold): đáp án gold giữ tên đúng mà bản chép nhiễu đã mất, nên mô hình học đoán.
- p95 của `60` lên 51 giây dù số lượt gọi và độ dài đầu ra không đổi. vLLM lần này chỉ có 4,09 GiB KV cache thay vì 4,53 GiB, nên có yêu cầu phải chờ và tỉ lệ trúng prefix cache còn 58% thay vì 66%. Đây là do máy lúc đo, không do mô hình.
- Nếu thử lại: nhiễu nhẹ hơn (5 dB, gần WER thật trên val), hoặc chỉ dùng bản chép nhiễu khi các thực thể trong call gold vẫn còn trong đó.

**Beam search 4 cho PhoWhisper-ft (configs 64, 65): không dùng**

| 100 mẫu đầu | WER greedy → beam 4 | thực thể đúng hệt | đầu-cuối với agent `40` | ASR giây/mẫu |
| --- | --- | --- | --- | --- |
| val_random (`54` → `64`) | 11,1 → 11,3 | 69,2 → 69,2 | 51,8 → 50,5 (−1,3 ± 1,6) | 0,54 → 2,34 |
| val_app (`55` → `65`) | 12,0 → 11,7 | 78,1 → 75,0 | 32,5 → 33,5 (+1,0 ± 1,2) | 0,66 → 2,33 |

Beam 4 đổi 19 và 28 bản chép trên 100 mà không nghe đúng hơn, còn chậm gấp 4. Giữ greedy.

**SFT trộn ASR lần 2 (configs 66-69): cũng không dùng**

Nhiễu 5 dB ra WER 9,9 trên cả `train_rest`, đúng mức ASR thật trên val; `--asr-keep-entities` đưa 40 trong 532 mẫu chép nhiễu về chữ đúng vì bản chép đã mất thực thể của call gold.

| | gold, val_random | gold, val_app | ASR, val_random | ASR, val_app | trung bình đầu-cuối |
| --- | --- | --- | --- | --- | --- |
| SFT (`40`, `41`, `54`, `55`) | 61,0 | 42,9 | 51,8 | 32,5 | 42,2 |
| SFT trộn ASR 5 dB (`66`-`69`) | 59,4 | 39,2 | 49,2 | 29,4 | 39,3 |
| hiệu | −1,6 ± 2,4 | −3,7 ± 2,3 | −2,6 ± 2,9 | −3,1 ± 2,2 | −2,9 |

Nghe đúng không còn tụt mạnh như lần 1, nhưng phần mất do ASR vẫn khoảng 10 điểm (−10,2 và −9,8): ở cả hai mức nhiễu, trộn bản chép nhiễu không làm agent chịu lỗi nghe tốt hơn. Bỏ hướng này.

**Máy tính ẩn `sys.calc` (configs 70-73): không dùng**

- Trên gold text, 17 mẫu của `40`/`41` sai vì ngày giờ hoặc số, mất 5 điểm trên val_random và 12 trên val_app. Phần lớn là phép tính chứ không phải đọc: 10 giờ trừ 7 phút đi đường, 15 lít nhân giá xăng, mục tiêu cũ cộng ngàn rưỡi bước, từ bảy rưỡi tới tám giờ mười là 40 phút.
- `src/lifemate/numbers.py` có máy tính (số; ngày giờ ISO cộng trừ 30m, 2h, 1d; hiệu hai giờ HH:MM ra số phút; chỉ nhận số học, không chạy mã) và bộ đọc số, giờ, thời lượng, thứ trong tuần của lời nói tiếng Việt. Bộ đọc làm việc trên chữ còn dấu, vì bỏ dấu thì "làm" trùng "lăm".
- Agent `calc: true` thêm `sys.calc` vào tập tool được phép và một dòng mô tả vào schema. Agent tự tính rồi trả kết quả như một tool, không gửi call này cho môi trường, nên nó không vào file nộp và không đụng điểm. Bước `sys.calc` không tính vào `max_steps`.
- `derive` tìm một phép tính một bước cho ra đúng giá trị gold từ các số đã có, chỉ khi giá trị không có sẵn và có ít nhất một toán hạng từ lời nói. Thứ tự thử: chép thẳng từ kết quả tool; mốc trong kết quả dời theo thời lượng nói ra; ngày trong bảng thời gian với giờ nói ra (không cần tính); rồi các mốc khác. Dấu cộng trừ theo "trước", "sớm" hay "sau", "muộn", "lùi". `build_sft.py` với config có `calc: true` chèn bước `sys.calc` trước call ghi tương ứng: 101 bước ở 64/1.018 mẫu `train_rest` (7 là số, còn lại là giờ). Với 17 mẫu lỗi ở trên, tìm được phép tính đúng cho 11.
- `tests/test_sft.py` phát lại hội thoại có `sys.calc` qua agent thật: mọi tin nhắn agent gửi trùng từng chữ với dữ liệu huấn luyện, file nộp không có `sys.calc`, state hash bằng gold.
- So `70`/`71` với `40`/`41` (nghe đúng), `72`/`73` với `54`/`55` (qua PhoWhisper-ft). `calc` huấn luyện trên Kaggle T4 (fp16, seed 0), nên mốc công bằng nhất cho phần nghe đúng là `t4-check` (`52`, `53`), chỉ khác ở các bước `sys.calc`.

| 100 mẫu | `val_random` | `val_app` |
| --- | --- | --- |
| gold, so với `t4-check` | 57,3 vs 58,0 (−0,7 ± 2,6) | 42,9 vs 41,9 (+1,0 ± 3,6) |
| gold, so với v1 (`40`, `41`) | −3,7 ± 2,4 | +0,0 ± 2,9 |
| PhoWhisper-ft, so với v1 (`54`, `55`) | 48,3 vs 51,8 (−3,5 ± 2,5) | 30,3 vs 32,5 (−2,3 ± 2,5) |

- **Mô hình gần như không gọi `sys.calc`**: 2 lần trong 400 lượt chạy, và cả hai mẫu đều 0 điểm. `train_0959` ("… + 1h20m") lỗi đọc biểu thức vì máy tính chưa nhận thời lượng ghép, lặp lại 2 lần; `train_1577` trừ hai giờ ngược chiều ra −240 phút.
- Chỉ 64/1.018 mẫu `train_rest` (6,3%) có bước `sys.calc` thật, nên mô hình học rằng hiếm khi cần gọi. Kể cả khi gọi đúng, trần lợi ích cũng nhỏ: 17 mẫu sai vì ngày giờ hoặc số, 11 tìm được phép tính.
- Bỏ cho v2: không nộp `full-calc` (config `93`). Nếu làm lại: nhận thời lượng ghép như "1h20m", và cần nhiều mẫu có phép tính hơn (dữ liệu tổng hợp) để mô hình thật sự gọi.

**Bản nộp v2 (configs 91, 94)**

Công thức của v1 (PhoWhisper-large-ft greedy, rồi Qwen3-4B SFT seed 0), agent huấn luyện lại trên cả 1.542 mẫu (`sft/train_full.jsonl`). `91` dùng `full` huấn luyện trên Kaggle (fp16), `94` dùng `full-local` huấn luyện ở máy nhà (bf16, `scripts/train_sft.py --data sft/train_full.jsonl --out models/qwen3-4b-sft-full-local`). Cả hai file nộp đủ 634 dòng, `lifemate validate` không báo lỗi.

- Không còn tập kiểm định nào chưa học, nên chỉ đo được trên public. **Điểm public: `91` 44,70, `94` 43,17** (v1 `90`: 42,15). Dùng `91` làm bản chính.
- Hai bản cùng dữ liệu và seed, chỉ khác máy và độ chính xác, mà trạng thái cuối đã khác nhau ở 159/634 mẫu (v1 so với `91`: 193). Theo `t4-check`, fp16 không kém bf16, nên 1,5 điểm giữa chúng là dao động huấn luyện, cỡ một lần đổi seed. Thêm 51% dữ liệu chỉ thêm khoảng 1,8 điểm (trung bình hai bản so với v1).
- Tăng ít vì phần thêm vào không chạm điểm yếu chính. 410/634 mẫu public (65%) cài ít nhất một trong 11 app không có trong train, và khoảng 260 dự đoán gọi các app đó; 524 mẫu thêm vào đều thuộc app đã thấy. Trên val, lượt SFT đầu tăng 26,8 điểm ở app đã thấy nhưng chỉ 6,8 ở app giữ lại. ASR giữ nguyên (PhoWhisper-ft chỉ học audio `train_rest`) và vẫn mất khoảng 10 điểm.
- Không nộp `92` (`full-rename-all`) và `93` (`full-calc`): đổi tên và `sys.calc` đều không thắng trên `train_rest`.
- Tab "Kiến trúc tốt nhất" của `app/compare.py` làm đúng các bước này từ `runs/` (gộp cả các git worktree nếu có): mỗi tầng một bảng xếp hạng, tổ hợp đang chọn, và bảng ghép có cột "mất do ASR".

**Hỏi lại khi bước đầu dừng ở app chưa thấy (configs 74-77): không dùng**

- Ở val_app, mọi ca "làm → đứng yên" đều là `stop` ngay bước đầu, không gọi tool nào: 10/100 với `41`, 13 với `drop` (`45`), 21 với `drop-rename-all` (`49`), tập sau chứa tập trước. Gỡ app giữ lại khỏi dữ liệu SFT không giảm lỗi này.
- `agent.recheck_stop` hỏi lại một lần khi bước đầu là `stop` mà máy có app chưa từng được gọi trong tập `known`. `nudge` liệt kê tool của các app đó và để mô hình tự chọn lại. `force` ép lượt sau phải gọi một tool của chúng hoặc `sys.ask_missing_info`.
- Chạy trên toàn bộ val (225 và 301 mẫu). Lượt LLM đầu tất định, nên mẫu được hỏi lại có điểm gốc đúng bằng điểm của dự đoán rỗng, còn mẫu không được hỏi lại trùng lượt gốc: 95/95 và 88/88 mẫu chung với `40`/`41` ra cùng call và cùng hash.

| | hỏi lại | val_random | val_app |
| --- | --- | --- | --- |
| `nudge` (`74`/`75`) | 8 / 20 mẫu | +0,00 | +0,33 |
| `force` (`76`/`77`) | 8 / 20 mẫu | −2,22 | +1,00 |

- Được nhắc thì mô hình gần như luôn dừng lại lần nữa (15/16 ca "làm" ở val_app). Bị ép gọi thì ở val_app chỉ 3/16 ca "làm" được điểm; 8 ca chuyển sang hỏi lại, 5 ca có gọi nhưng vẫn 0 điểm. Ở val_random, 5 mẫu gold rỗng bị phá. Mô hình dừng không phải vì thiếu một cú hích: nó không đọc được schema của app chưa thấy. Hướng sửa là dạy trên nhiều app hơn (đề xuất 5).

**Mồi tên riêng cho PhoWhisper-ft (configs 29_\*-hint, 78, 79): lợi nhỏ, chưa chắc**

| 100 mẫu chung với `54`/`55` | WER | thực thể đúng hệt | đầu-cuối với agent `40` | ASR giây/mẫu |
| --- | --- | --- | --- | --- |
| val_random (`54` → `78`) | 11,1 → 11,2 | 69,2 → 71,8 | 51,8 → 52,7 (+0,9, 95% [−1,3; +3,5]) | 0,49 → 1,02 |
| val_app (`55` → `79`) | 12,0 → 11,6 | 78,1 → 75,0 | 32,5 → 34,0 (+1,5, 95% [−0,8; +4,5]) | 0,58 → 1,10 |

Mồi đổi 24 và 33 bản chép trên 100; mỗi tập 3 mẫu tốt lên, 1 mẫu tệ đi. Có mồi thì mỗi mẫu chạy lẻ nên ASR chậm gấp đôi, vẫn khoảng 1 giây. Cần chạy `54`/`55` trên toàn bộ val để so 526 mẫu trước khi đổi bản nộp.

**Độ trễ và giới hạn thời gian (config 95)**

- Bản nộp `91` chạy tuần tự (`workers: 1`) trên 200 mẫu public ngẫu nhiên: p50 3,95 giây, p95 7,21, chậm nhất 10,09 (gồm ASR lấy từ cache). Mỗi lượt LLM trung bình 1,07 giây, chậm nhất 3,73; nhiều nhất 8 lượt một mẫu.
- Cùng 200 mẫu đó khi `91` chạy 8 mẫu song song: p95 27,6, chậm nhất 46,9, có 8 mẫu quá 30 giây. Hai lượt ra cùng call ở 200/200 mẫu. Mẫu quá 30 giây là do chờ nhau, không phải do sinh dài.
- `agent.time_budget` (giây mỗi mẫu, tính cả ASR): trước mỗi lượt LLM còn dưới `min_step_sec` thì dừng, mỗi lượt chỉ được phần còn lại và không thử lại; call đã gọi vẫn được nộp. Khi chạy tuần tự nó không cắt mẫu nào. Chỉ nên bật trong config chạy Docker, không đặt trong `base.yaml`, vì các lượt đo 8 worker sẽ bị cắt và lệch so với kết quả cũ.

**Luật viết tay sửa tham số call ghi (`src/lifemate/fixes.py`, configs 88, 89, 96-98)**

Những chỗ mô hình 4B hay sai mà đọc được thẳng từ lời nói thì để mã làm, ở tầng D, trước khi call chạy. Mỗi luật là một cờ trong `guards:`, mặc định tắt.

| Luật | Làm gì | Ví dụ trên val |
| --- | --- | --- |
| `clock_fix` | Ngày giờ ISO của call ghi lệch ngày giờ nói ra: sai sáng/chiều, hoặc lời chỉ có một giờ, một ngày mà call khác | "9 giờ kém 15 tối mai" ra 18:45 → 20:45 |
| `money_k` | Số tiền VND dưới 1.000 nhân 1.000 (trừ quy đổi ngoại tệ) | ASR "bốn mươi hai nhìn" ra 42 → 42.000 |
| `span_min` | Tham số phút khi lời có "từ A tới B" | "từ bảy rưỡi tới tám giờ mười": 60 → 40 |
| `ask_missing` | Hỏi lại thay vì bịa: số tiền khi lời chỉ nói "chút tiền", giờ hẹn khi không ai nói giờ | "chuyển ví cho Minh chút tiền": 100.000 → hỏi lại |
| `pick_one` | "Mấy gói … hủy bớt một gói thôi": hỏi lại mục nào thay vì chọn bừa | |

- Bộ đọc mới hiểu buổi đứng trước giờ ("tối mai bảy giờ"), "kém", câu tự sửa ("chín giờ, à nhầm, chín rưỡi"), "thứ ba, hôm hai chín", "chủ nhật mười một tháng mười", tuần tính từ thứ Hai như bảng thời gian. Trên call gold của cả train, lời chỉ có một ngày thì ngày gold trùng ở 346/351 tham số chính, chỉ có một giờ thì 151/151.
- Không đụng giá trị có nguồn: có trong lịch sử hay kết quả tool (kể cả giờ viết trong tin nhắn, ghi chú), hay tính được một bước từ đó (`numbers.derive`). Không sửa khi lời có dời theo thời lượng ("trước 15 phút", "hai tiếng sau"), mốc khác ("hôm đó", "đúng giờ", "kịp bảy giờ", "trước mười một giờ").
- Bật mọi luật rồi chạy call gold của cả 1.545 mẫu train: không call nào đổi, state không đổi (`tests/test_fixes.py`).
- Câu hỏi lại được hoãn tới khi agent xong, và call bị hoãn trả `ok: true` kèm ghi chú. Gửi ngay thì môi trường trả `waiting_for_user`, còn trả `ok: false` thì agent SFT cũng dừng luôn, bỏ phần còn lại (train_0352, public_0033 mất lệnh đặt báo thức). Cái giá: ở train_1296 agent làm tiếp bằng một lệnh tắt đèn thừa (điểm vẫn 0 như trước).

Chạy thật trên toàn bộ val, ghép cặp với cùng mô hình `qwen3-4b-sft`:

| | không luật | có luật | hiệu | tốt lên / tệ đi |
| --- | --- | --- | --- | --- |
| val_random, nghe đúng (`86` → `88`) | 59,17 | 61,24 | +2,07 ± 0,93 | 5 / 0 |
| val_app, nghe đúng (`87` → `89`) | 47,58 | 48,25 | +0,66 ± 0,47 | 2 / 0 |
| val_random, PhoWhisper-ft có mồi (`78` → `96`) | 51,95 | 53,29 | +1,33 ± 0,77 | 3 / 0 |
| val_app, PhoWhisper-ft có mồi (`79` → `97`) | 36,15 | 36,48 | +0,33 ± 0,33 | 1 / 0 |

- Ngoài các mẫu luật nổ, mọi mẫu khác ra đúng call cũ: lượt LLM tất định, luật không đổi gì thì không có gì khác. Vì vậy 6 mẫu có hỏi lại được chạy lại riêng với cách trả `ok: true` rồi ghép vào (`88c`, `96c`, `97c`); chỉ train_0352 đổi, 0 → 1.
- Lợi nhỏ (khoảng +1 điểm) nhưng không mẫu nào tệ đi. Luật được dựng sau khi xem lỗi của chính các tập val này, nên trên public có thể ít hơn. Trên dự đoán public của `91`, luật nổ ở khoảng 14/634 mẫu; đã đọc từng chỗ và siết luật đến khi những chỗ còn lại hợp lý hoặc nằm ở mẫu ASR đã hỏng sẵn ("mỗi cái một vé thôi" là số lượng, "bán bớt một trăm cổ" không phải chọn một trong nhiều, giờ nhắc tính từ thời gian nấu không bị sửa).
- `98` là bản nộp `91` cộng các luật này, 7 mẫu có hỏi lại chạy lại như trên (`98c`) rồi ghép: file nộp hợp lệ, khác `91` ở 11/634 mẫu. 5 mẫu sửa tham số (báo thức "năm rưỡi sáng mai" 17:30 → 05:30, "bảy giờ kém mười lăm sáng mai" 07:45 → 06:45, hai ngày "chiều nay", "trưa nay" bị lệch, "ba trăm rưỡi" 350 → 350.000), 6 mẫu hỏi lại (ba câu "ít tiền", "mai gọi mình dậy" không có giờ, hai mẫu ASR hỏng). `pick_one` từng nổ nhầm ở "con mở một cái thôi" (mở mới, không chọn trong cái có sẵn); nay cần động từ hủy, tắt, xoá, bỏ, dừng đứng trước. p95 27,6 giây với 8 mẫu chạy cùng lúc, như `91`.
- Đã thử mà không dùng: `id_guard: block` chạy lại offline trên 4 lượt trên đổi 1/1.052 mẫu.

## 9. Mô hình và giấy phép

| Mô hình | Tham số | Giấy phép | Ngày công bố | Ghi chú |
| --- | --- | --- | --- | --- |
| [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | 0,81 tỷ | MIT | 10/2024 | |
| [vinai/PhoWhisper-large](https://huggingface.co/vinai/PhoWhisper-large) | 1,55 tỷ | BSD-3-Clause | 12/2023 | |
| [Qualcomm-AI-Research/PhoASR-whisper-small](https://huggingface.co/Qualcomm-AI-Research/PhoASR-whisper-small) | 0,24 tỷ | BSD-3-Clause-Clear và Qualcomm Responsible AI License | 07, theo lịch sử commit | Cần đọc điều khoản Qualcomm RAI. Model card cảnh báo ít dữ liệu giọng miền Trung |
| [khanhld/chunkformer-ctc-large-vie](https://huggingface.co/khanhld/chunkformer-ctc-large-vie) | 0,11 tỷ | **CC BY-NC 4.0** | 02/2025 | Phi thương mại: hỏi BTC trước khi đưa vào bài nộp |
| [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | 4,0 tỷ | Apache-2.0 | 07/2025 | |
| Phonetic Modeling of Dialectal Variation in Vietnamese Speech ([arXiv 2605.24451](https://arxiv.org/abs/2605.24451)) | | | 05/2026 | Chưa công bố mã. Ý tưởng gộp âm theo phương ngữ có bản luật tay ở `phonetic_entity_fix` |
