# NOTES.md — phiên Claude Max LM-16 (08/11/2026, 03:40–06:45 giờ Việt Nam)

Mục tiêu: nâng public từ 45,92 (config 98) trong đúng thể lệ. Máy này chỉ có CPU + RTX 3050 4 GB: không chạy ASR/LLM,
mọi số đo là **replay** trên vết đã lưu (`runs/` trong gói dữ liệu), bằng `tools/replay_score.py` của nhánh
`pro/max-kickoff` cộng `scripts/replay_guards.py` (mới). Số ghi "replay" là đo thật trên CPU; số ghi "ước lượng" là chưa đo.

## 1. Công cụ đo đã kiểm chứng

`scripts/replay_guards.py` chạy lại các call LLM *đã đề nghị* (`guard_log[].proposed`) qua `GuardedEnv` với guards tuỳ
chọn trên ToolEnv mới, rồi chấm bằng `score_sample`. Kiểm chứng: guards của config 98 (clock_fix, money_k, span_min,
ask_missing, pick_one) trên vết 86/87 (text gold, không luật) cho **+2,07 / +0,66**, trùng số lượt GPU 88/89 so với 86/87
trong docs/BASELINE.md. Vậy có thể tin replay cho các luật tất định ở tầng D.

| Guards (replay) | 54n N-best val_random | 55n N-best val_app | 54f greedy | 55f greedy | 86 gold | 87 gold |
| --- | --- | --- | --- | --- | --- | --- |
| như lượt chạy (không luật) | 56,38 | 39,33 | 51,86 | 35,85 | 59,17 | 47,58 |
| luật của 98 | 57,27 (+0,89) | 39,66 (+0,33) | 53,20 (+1,33) | 36,18 (+0,33) | 61,24 (+2,07) | 48,25 (+0,66) |
| luật của 98 + ask_ambiguous | **57,71 (+1,33)** | **39,99 (+0,66)** | 53,64 (+1,78) | 36,51 (+0,66) | 61,68 (+2,52) | 48,58 (+1,00) |

Không mẫu nào tệ đi ở dòng cuối (entity_pick tắt). 98n (N-best + luật 98) ≈ 57,27 / 39,66; cộng ask_ambiguous ≈ 57,71 / 39,99.
Bộ test đầy đủ trên nhánh max/entity-guards: xem commit message / PR (pytest -q tests, Python 3.12, gói dữ liệu BTC).

## 2. Nhánh đã làm

### max/config98-nbest (PR 1)
- Merge `NhatNguyen/asr-ft-v2` (daf2ff8) vào config98 (b9d8a32): hai nhánh cùng gốc 45e88ce, chỉ chung `asr/base.py`,
  git tự gộp sạch (cache dir env var + mồi từ vựng app). Không có xung đột ở agents/guards/fixes như brief lo.
- `configs/experiments/98n_submit_public_full_fixes_nbest4.yaml` = 98 + `asr: {num_beams: 4, n_best: 4, batch_size: 2}`,
  `agent.show_alts: true`. `configs/docker-nbest.yaml` = docker.yaml trên 98n. `n_best: 1` thì mã và prompt y như 98
  (HFWhisper chỉ rẽ sang `_nbest` khi n_best > 1; alts rỗng thì user prompt không đổi) — có test.
- Docker: `ARG/ENV CONFIG` trong Dockerfile, `CONFIG=` trong `scripts/docker_build.sh`; predict.sh đã đọc biến CONFIG.
  `CONFIG=configs/docker-nbest.yaml TAG=lifemate:98n scripts/docker_build.sh`. Không đặt gì → image như cũ.
- Độ trễ (ước lượng, chưa đo): beam 4 trong 64/65 làm ASR 0,6 → 2,3 s/mẫu (batch 16, RTX 5060 Ti); N-best batch 2 có thể
  3–4 s/mẫu, p99 cả mẫu khoảng 11–13 s thay cho 9,0 s, còn xa 25 s. A30 24 GB có thể thử `batch_size: 4`.
- Test: `tests/test_nbest_config.py` (5 test) + bộ cũ.

### max/entity-guards (PR 2, tách từ config98)
- `src/lifemate/entities.py` + hai cờ guards (mặc định tắt): `ask_ambiguous`, `entity_pick`.
- `ask_ambiguous` (bật được): call ghi dùng ID từ danh sách vừa đọc trong lượt này có ≥ 2 mục, **không mục nào có chữ nào
  trong lời (lượt này + các lượt user trước)**, lời lượt này không có dấu hiệu phân biệt (số nhiều, "đó/ấy/khác", so sánh,
  trạng thái, giờ/ngày khi mục có thời gian, chữ số khi tên mục có số), danh sách có tên tiếng Việt → hoãn thành
  `sys.ask_missing_info(tool, [id_param])` ở cuối (cơ chế của ask_missing). Replay: +0,44 / +0,33 ở cả ba loại đầu vào,
  0 mẫu tệ đi (train_1103, train_1713). Gold 1.545 mẫu không đổi call nào (test).
- `entity_pick` (để tắt): đổi ID khi chữ khớp của mục đã chọn là tập con thật của chữ khớp mục khác với từ riêng.
  Replay: 0 mẫu tốt lên, 1 tệ đi (train_0478, ASR sai "máy rửa bát" → "máy của bác"). Bản thô hơn (không xét lời lượt
  trước, không xét dấu hiệu, đổi khi mục đã chọn không có *từ riêng*) làm **giảm 0,4–1,8**; ask thô giảm 0,6–3,5. Ghi lại
  để không ai thử lại kiểu đó.
- `scripts/replay_guards.py`: công cụ đo ở mục 1.

### max/agent-ask-guard (PR 3, tách từ config98)
- Cờ `agent.ask_guard` (tắt mặc định): LLM gọi `sys.ask_missing_info` cho tham số có x-ref khi chưa đọc tool nguồn của app
  đó (và lịch sử chưa có app đó) → không ghi câu hỏi, nhắc "đọc <tool> trước rồi chọn" một lần mỗi tool
  (`GuardedEnv.unread_sources`). Configs 54a/55a (A/B với 54/55), 98a. Test LLM giả: `tests/test_ask_guard.py` (6).
- Bằng chứng: 7/17 ca "làm → hỏi lại" trên val_app hỏi trước khi đọc (trần +2,3 val_app; ước lượng +0,5…+1,5).
  Không đo được bằng replay (đổi hành vi giữa episode) → Codex chạy 54a/55a.

### max/synth-nbest (PR 4, tách từ config98; gộp synth d5ca6dd rồi gộp max/config98-nbest)
- Synth chỉ chung docs với config98 (một xung đột `docs/roadmap.yaml`, giữ bản synth có số đo); N-best gộp sạch.
- Configs: `99n_*` (= 54n/55n với checkpoint synth → synth × N-best đầu-cuối trên val, so ghép cặp 54n/55n), `98s`
  (= 98n với `models/qwen3-4b-sft-full-synth`), `configs/docker-synth-nbest.yaml`; `kaggle/run_sft.py` thêm `full-synth`.
- Bộ sinh synth thêm loại `pick_detail` (6/108): hai mục cùng tên gốc ("iPad của bé Na cũ" / "... mới"), người dùng nói
  đủ chi tiết → gold đọc rồi ghi, **không hỏi**; đối trọng của `ask_ambiguous` (luôn hỏi) vì synth làm "hỏi lại thừa"
  tăng ở 3/4 seed (BASELINE). Test: gold của 300 mẫu phát lại đúng kết quả/state, 120 mẫu khớp từng chữ ReactJSON.
- Chưa có số đo (đổi checkpoint, đổi dữ liệu): Codex chạy 99n; Kaggle train full-synth (`build_sft.py --subset all --synth 600`).
  Ngưỡng: 99n val_app ≥ 54n/55n + 2, val_random ≥ −1; public 98s ≥ 98n + 2.

### Trạng thái test (CPU, Python 3.12, gói dữ liệu chỉ có split train)
- max/config98-nbest: 87 passed, 1 failed; max/entity-guards: 100 passed, 1 failed; max/agent-ask-guard: 86 passed, 1 failed;
  max/synth-nbest: 97 passed, 1 failed. CI GitHub (uv, ruff, pytest không dữ liệu) xanh ở cả bốn PR (#8, #9, #10, #11).
  Lỗi duy nhất ở cả ba nhánh là `tests/test_fixes.py::test_missing_asks_for_an_amount_or_a_time_nobody_said`: cần
  `public_test/public_test_tools_catalog.json`, không có trong gói dữ liệu của máy này (lỗi có sẵn, không do thay đổi).

## 3. Kết quả âm khác (đo replay, không làm)
- Ép mọi call đọc về `{}` (luật 5 của prompt): −4,5 val_random vì gold `contacts.search` luôn có `query`; chỉ vài tool
  (calendar.list_events 83 %, smart_home.get_status 87 %) mới hay dùng `{}`, và LLM đã làm đúng phần lớn.

## 4. Chẩn đoán: vì sao 46 còn top 96,5 (bằng chứng từ vết 55n val_app, 301 mẫu, 202 sai)
- Với transcript đúng 100 % agent chỉ đạt 59,17 / 47,58: ASR chỉ chiếm 3–8 điểm, phần còn lại là quyết định của LLM.
- Lỗi quyết định trên val_app: làm → đứng yên 19, thiếu lệnh ghi 23, sai tool ghi 20, làm → hỏi lại 17, hỏi lại → làm 16
  = 95/301 (32 %). Gần như toàn bộ rơi vào app giữ lại (heldout): mô hình SFT trên train_rest *không bao giờ thấy app đó
  được gọi* nên dừng, chọn tool của app quen, hoặc hỏi lại thay vì đọc. Public có 410/634 mẫu dùng app không có trong
  train, nên public (45,9) giống val_app (39) hơn val_random (56). Đây là nút thắt chính, không phải tầng guard.
- Hash sai ở 50 % val_app; F1 khi hash đúng vẫn 78 (fuzzy, thừa/thiếu đọc).
- Luật tất định ở tầng D chỉ lấy lại được 1–2,5 điểm (mục 1): trần của hướng "thêm guard" đã gần.
- Điểm theo app giữ lại trên val_app (55n N-best / 87 text gold), thấp nhất trước: loyalty 17,8 / 14,4 (làm → đứng yên 6,
  làm → hỏi lại 6: mô hình không biết get_points → redeem(program=prog_…, reward_id=rw_…)); package_tracking 16,7 / 42,0
  (reschedule_delivery thay cho schedule_pickup, sai ngày); video_meeting 34,2 / 46,9; library 37,8 / 50,8; learning
  40,2 / 37,1 (sai tool ghi 6–8: calendar thay cho schedule_lesson / log_progress); medication 40,6 / 40,7; appliances
  42,0 / 56,6; subscriptions 46,5 / 58,7; fitness 47,7 / 54,2; door_lock 59,7 / 71,6. Synth app lạ nên ưu tiên các mẫu
  "đọc → ghi với ID từ kết quả đọc" cho loyalty, learning, package_tracking, video_meeting.
- `ask_tie` (hỏi khi mục đã chọn và mục khác khớp *cùng* tập chữ trong lời, kiểu "gọi cho Minh" có hai Minh): replay
  0 tốt / 1–2 tệ mỗi lượt (train_0280, 0639, 0991) và đổi 5 call gold → bỏ, không giữ mã. Trong gold train, 114/217 mẫu
  hỏi lại là sau một call đọc về ≥ 2 mục, nhưng chỉ 5 ca là "không mục nào có chữ trong lời"; phần lớn còn lại (73 ca
  contact_id) phân biệt bằng ngữ cảnh mà chữ không bắt được → phải dạy bằng dữ liệu, không bằng luật.
- Chép `content`/`body` từ mệnh đề sau "là/rằng" trong lời (không vào hash nên replay đo chính xác): trên 4 lượt sửa được
  0–1 cặp, phá 1–5 cặp đang trên ngưỡng 0,6 (gold thường thêm xưng hô, tên gói, giờ: "Chị Hoa ơi, cửa trước khoá rồi
  chị nhé"). LLM đạt > 0,6 ở 24/32, 23/48, 29/36, 30/52 cặp; phần còn lại là diễn đạt, không có luật tất định. Bỏ.
- Luật xưng hô cho `contacts.search` (thêm "Cô/Thằng" trước query): 0 mẫu sửa được trên 4 lượt (10–11 query lệch mỗi lượt
  đều lệch kiểu khác), bỏ.
- Các ca "hỏi lại → làm" có quy ước rõ: đáp án hỏi khi danh sách đọc về có ≥ 2 mục mà lời không phân biệt. Mô hình
  không học được quy ước này vì dữ liệu SFT ít (217 mẫu hỏi lại / 1.545). Nên sinh thêm mẫu hỏi lại trong synth.

## 5. Đề xuất cho vòng sau (có lệnh)
1. Chạy GPU hai config để chốt PR 1 + PR 2 (Codex, RTX 5060 Ti; ASR trước, vLLM sau như 54n/55n):
   `lifemate run configs/experiments/54n_... --set guards.ask_ambiguous=true --set name=54q` và 55q tương ứng; chấp nhận
   nếu ghép cặp không tệ hơn 54n/55n (replay dự đoán +0,44 / +0,33) và không mẫu nào mất điểm vì hỏi lại thừa.
   Rồi `CONFIG=configs/docker-nbest.yaml TAG=lifemate:98n scripts/docker_build.sh` + `docker_check.sh`, chấp nhận p99 < 20 s,
   hai lượt trùng byte.
2. Việc cho điểm lớn (cần train, Kaggle): synth cho **app chưa thấy** với tỉ trọng hỏi lại đúng quy ước trên; đo trên val_app
   (text gold rồi N-best). Mục tiêu chấp nhận: val_app gold ≥ 50 (87 đang 47,58), val_random không giảm quá 1.
3. Vòng agent (cần GPU, có thể làm sau cờ): khi LLM gọi `sys.ask_missing_info` cho tham số có x-ref mà tool đọc nguồn của app
   đó chưa được gọi trong lượt và ID chưa có trong lịch sử → trả cho LLM ghi chú "đọc <tool> trước" một lần thay vì ghi
   nhận câu hỏi. Bằng chứng: 7/17 ca "làm → hỏi lại" trên val_app hỏi trước khi đọc (train_1121, 1203, 1365, 1371, 1450...).
   Chưa cài vì không đo được bằng replay; đề xuất Codex cài + đo A/B.
