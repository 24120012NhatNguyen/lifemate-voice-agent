# Docker cho Qwen3.5-4B chỉ văn bản: cấu hình và bảng ngưỡng tính trên giấy · LM-50 phần B

Theo `BRIEF_LM50.md` bản 11/11 17:30: **hiện không chạy Docker trên GPU**. Tài liệu này chỉ gồm cấu hình đã viết, các ngưỡng của thể lệ tính từ số đã đo sẵn, và danh sách việc phải chạy thật vào ngày chủ dự án cho phép. Mỗi con số ghi **[đo]** kèm nguồn, hoặc **[ước]** kèm cách tính. Máy đo là RTX 5060 Ti 16 GB (gọi là "máy nhà"), trừ chỗ ghi khác.

Kết luận ngắn:

| Ngưỡng | Kết quả trên giấy | Còn phải kiểm bằng lượt thật |
|---|---|---|
| Tham số | Đạt: 5.749.056.256 < 6 tỷ; LLM 4.205.751.296 < 5 tỷ | Không |
| Ngữ cảnh 16.384 | Đạt: lượt gọi dài nhất 6.867 token vào + 400 token ra = 44% cửa sổ | Không |
| 30 giây mỗi mẫu | Không thể vượt theo thiết kế (cắt ở 25 giây). Máy nhà, Qwen3.5 tuần tự đã đo: p99 11,5 giây, chậm nhất 13,0. A30 theo thông số công bố: mẫu chậm nhất từ 6,3 tới 30,0 giây; ở đầu xấu nhất 9 trên 1.686 mẫu (0,5%) chạm mức cắt, không mẫu nào quá 30 | Có: ASR N-best đo riêng từng mẫu, và một lượt trên GPU của BTC nếu có |
| Chạy lại ra cùng kết quả | Trên lý thuyết chế độ tuần tự lặp lại được; chưa có lượt nào chứng minh. Chế độ 8 luồng đã đo lệch 0,0033 và 0,0000 PA giữa hai lượt, dưới ngưỡng 0,03 của BTC | Có: hai lượt tuần tự so từng byte |

## 1. Cấu hình đã viết

- `configs/docker-qwen35.yaml`: bản nộp 98q (N-best 4, 5 luật + 3 luật chuẩn hoá, `enable_thinking: false`) chạy `workers: 1`, `time_budget: 25`.
- `docker/predict.sh`: đọc `config.json` của model; gặp `model_type: qwen3_5_text` thì phục vụ với `--gdn-prefill-backend triton --max-num-seqs 1 --max-num-batched-tokens 8192 --chat-template prompts/qwen35_scaffold.jinja`, `GPU_UTIL=0.80`, không bật `DETERMINISTIC`. Model Qwen3 chạy y như trước. Thêm `SUBSET`, `EXTRA_SET`, `MAX_NUM_SEQS`, `LLM_DETERMINISTIC`, `VLLM_EXTRA` để đổi chế độ khi kiểm.
- `scripts/docker_check.sh`: nhận tập val của train (`SPLIT`, `SUBSET`, `LIMIT`), model báo cáo theo `ASR` / `LLM`, đo ASR từng mẫu theo đúng cách nghe (`ASR_BEAMS`, `ASR_NBEST`), `DOCKER_ARGS`.
- `scripts/docker_report.py --params-only`: bảng tham số từ header safetensors.
- `scripts/docker_cpu_check.sh`: kiểm một image chỉ bằng CPU (không `--gpus`, không khởi động vLLM). **Chưa chạy**: viết sau lệnh dừng Docker.
- Dựng image: `LLM=models/qwen35-4b-sft-<tên> CONFIG=configs/docker-qwen35.yaml TAG=lifemate:q35 scripts/docker_build.sh`.

Đã kiểm không cần GPU:

- **[đo]** `bash -n` ba script; `lifemate.config.load("configs/docker-qwen35.yaml")` ra `workers 1`, `time_budget 25`, `temperature 0.0`, `seed 0`, ASR `num_beams 4, n_best 4, batch_size 2`.
- **[đo]** Chat template: với tokenizer của `models/qwen35-4b-sft-synth` (`Qwen3_5Tokenizer`), `qwen35_scaffold.jinja` kết thúc prompt bằng khung think rỗng và trùng template gốc ở bước đầu (`enable_thinking=False`).
- **[đo]** vLLM 0.33.0 có cờ `--gdn-prefill-backend {flashinfer,triton,cutedsl,aiter_flydsl,sycl}`.
- **[đo]** Image `lifemate:q35` (model 161) đã dựng lúc 15:50 trong 4 phút 35 giây, 48,5 GB. Lúc 15:51 một lệnh CPU trong image xác nhận `CONFIG`, `LLM_MODEL`, `ASR_MODEL`, `model_type qwen3_5_text`, template, config và `gcc` đều có.

Đã chạy trước lệnh dừng (17:30) và không lặp lại: một lượt 4 mẫu val_app trong container tắt mạng lúc 16:36–16:39, chạy trọn ba pha, file nộp hợp lệ. Hai lượt đo 150 mẫu (tuần tự, 8 luồng) bị huỷ khi còn chờ khoá; **chưa có lượt Docker nào đo tất định hay phân bố độ trễ của Qwen3.5**.

## 2. Tham số

**[đo]** Đếm từ header safetensors bằng `docker_report.py --params-only` trên PC:

| Model | Tham số | Tensor LoRA | Tensor vision |
|---|---:|---:|---:|
| `models/phowhisper-large-ft` (ASR) | 1.543.304.960 | 0 | 0 |
| `models/qwen35-4b-sft-synth` (LLM chỉ văn bản, LoRA đã gộp) | 4.205.751.296 | 0 | 0 |
| **Tổng** | **5.749.056.256** | | |

- So với 6 tỷ: còn 250.943.744. LLM so với 5 tỷ (ràng buộc ghi trong `docs/METHOD.md`): còn 794.248.704.
- Cách FAQ của BTC đếm (`fable-work/ref/faq.txt`, mục cách tính tham số): (c) Qwen3.5-4B tính là 4B; (d) không tính vision encoder trong LLM với đề LifeMate; (e) vẫn tính tham số LoRA. Theo cách đó là 4,0 tỷ + 1,543 tỷ = 5,543 tỷ.
- Bản nộp không dựa vào (d): vision (333.514.240) và MTP (120.599.552) đã bị bỏ khỏi file trọng số, theo `export_receipt.json`. Nếu để nguyên checkpoint gốc 4.659.865.088 thì tổng đếm thô là 6.203.170.048, vượt 6 tỷ.
- LoRA đã gộp vào trọng số nên không cộng thêm. Để riêng thì phải cộng 32.464.896 (dòng `trainable params` của trainer).
- Model train trên dữ liệu khác (169, bản full) có cùng kiến trúc nên cùng số tham số; chạy lại `--params-only` trên đúng thư mục model nộp.

## 3. Ngữ cảnh

**[đo]** `prompt_tokens` lớn nhất của một lượt gọi LLM, đọc từ `samples.jsonl` (`steps[].prompt_tokens`):

| Lượt | Mẫu | `prompt_tokens` lớn nhất | Lượt gọi nhiều nhất mỗi mẫu |
|---|---:|---:|---:|
| 161 val_app (`runs/20261111-094101_…`) | 301 | 6.867 | 8 |
| 161 val_random (`runs/20261111-094618_…`) | 225 | 6.424 | 7 |
| 169 val_app (`runs/20261111-162558_…`) | 301 | 6.867 | 8 |
| 169 val_random (`runs/20261111-163122_…`) | 225 | 6.654 | 6 |
| 98q public (`runs/20261111-101430_…`) | 634 | 6.652 | 8 |
| 98c public | — | chưa có lượt trong `pro-threshold/runs` lúc 17:39 | — |

Cửa sổ là `--max-model-len 16384`, `max_tokens: 400`. Trường hợp xấu nhất 6.867 + 400 = 7.267 token, 44% cửa sổ, còn 9.117 token. Ngữ cảnh do schema của các app trong mẫu quyết định; private có app mới nên biên này là thứ bảo vệ.

## 4. Độ trễ mỗi mẫu

Luật (FAQ LifeMate câu 1): 30 giây tính từ lúc nhận đầu vào, gồm cả nhận dạng giọng nói, tới khi có kết quả; không tính nạp model.

### 4.1. Vì sao không thể vượt 30 giây

`time_budget: 25` tính cả ASR. Trước mỗi lượt gọi LLM agent tính thời gian còn lại, hết thì dừng và nộp các call đã gọi; mỗi lượt gọi mang `timeout` bằng đúng thời gian còn lại, nên một lượt đang chạy cũng bị cắt (`src/lifemate/agent/agents.py` dòng 304–314). GPU chậm hơn không làm vượt hạn; nó làm một số mẫu bị cắt và mất điểm ở các mẫu đó. Câu hỏi thật là: **bao nhiêu mẫu chạm 25 giây**.

Một điểm chưa chắc: `sec_asr` của lượt nộp là phần chia đều của ASR chạy theo lô, không phải thời gian nghe riêng của mẫu đó. Biên 5 giây giữa 25 và 30 là để phủ chênh lệch này; phép đo ASR từng mẫu với N-best 4 chưa có (xem 6).

### 4.2. Số đã đo trên máy nhà

| Ký hiệu | Lượt | p50 | p95 | p99 | Chậm nhất |
|---|---|---:|---:|---:|---:|
| M1 | Qwen3.5 tuần tự trong Docker, 4 mẫu val_app, 16:39: agent | 3,04 | — | — | 3,60 |
| M1 | cùng lượt: cả mẫu (ASR chia đều 3,20 gồm cả nạp model ASR) | 6,24 | — | — | 6,79 |
| M2 | Qwen3.5 ở 8 luồng, 5 lượt, 1.686 mẫu: agent | 7,4–8,2 | 15,8–17,5 | 19,3–29,9 | 22,0–33,4 |
| M3 | Qwen3-4B tuần tự trong Docker, N-best 4, 634 mẫu public, hai lần kiểm 8/11 và 9/11: agent | 3,15–3,21 | 7,06–7,10 | 8,67–8,72 | 9,95–10,33 |
| M3 | cùng hai lần: cả mẫu (ASR chia đều) | 6,07–6,27 | 11,36–11,59 | 12,62–13,13 | 14,57–18,55 |
| M4 | ASR N-best 4, phần chia đều mỗi mẫu, trong các lượt 161 / 169 / 98q | 1,81–1,99 | 3,96–4,58 | 3,96–6,75 | 6,57–7,23 |
| **M5** | **Qwen3.5 (161) tuần tự, `workers: 1`, 100 mẫu val_app, lượt của LM-49 lúc 17:52: agent** | **2,69** | **4,26** | **9,93** | **10,64** |
| **M5** | **cùng lượt: cả mẫu (ASR chia đều)** | **4,69** | **9,16** | **11,46** | **13,04** |
| M5 | cùng lượt: một lượt gọi LLM (306 lượt) | 0,92 | 1,60 | 1,93 | 3,32 |

Nguồn: M1 `fable-work/out/LM-50/docker_check/1111-1625_tiny_val_app_4/run1/report.json`; M2, M4 `samples.jsonl` của năm lượt ở mục 3; M3 `fable-eval/pr-8-5c3f7e3/artifacts/docker_check/20261108-083546_docker/run1/report.json` và `fable-eval/pr-11-bd2b821/artifacts/docker_check/20261109-181751_docker/run1/report.json`; M5 `fable-eval/lm49-max-vote-repro-dbe2e04/runs/20261111-175219_161_qwen35-sft_synth_val-app_latency/` (config `161_…_latency.yaml` của LM-48: 161 val_app với `workers: 1`, `limit: 100`).

M5 chạy ngoài Docker, server vLLM phục vụ như lượt đo (`--max-num-seqs 8`, `DETERMINISTIC=0`) nhưng chỉ có một client, nên mỗi lúc cũng chỉ một chuỗi. Nó là số đo tuần tự của đúng model và đúng pipeline; khác bản nộp ở chỗ không qua container và ASR lấy từ cache (phần chia đều theo lô).

Tải của agent **[đo]**, 1.686 mẫu: trung bình 3,06–3,21 lượt gọi mỗi mẫu, nhiều nhất 8; token ra mỗi mẫu p50 80–93, p99 242–268, nhiều nhất 465; một lượt gọi ra nhiều nhất 120 token.

Ghi chú về M2: ở 8 luồng một mẫu phải chia GPU với bảy mẫu khác, nên M2 là trần của thời gian tuần tự chứ không phải thời gian tuần tự. Mẫu chậm nhất của M2 (`train_1618`, 33,4 + 4,4 giây) không nói lên điều gì về chế độ nộp.

### 4.3. Độ trễ tuần tự của Qwen3.5 trên máy nhà

**[đo]** M5: cả mẫu p50 4,69, p95 9,16, p99 11,46, chậm nhất 13,04 giây trên 100 mẫu val_app. Không mẫu nào quá 20 giây. Qwen3.5 tuần tự nhanh hơn Qwen3 tuần tự (M3) ở trung vị (agent 2,69 so với 3,15–3,21).

**[đo]** Khớp thời gian 306 lượt gọi LLM của M5 theo `giây ≈ a + b · token mới + c · token ra` (token mới là phần prompt chưa nằm trong prefix cache của mẫu): a = 0,071 giây, b = 0,136 ms mỗi token mới (khoảng 7.350 token/giây prefill), c = 22,03 ms mỗi token ra (45 token/giây). Cộng theo mẫu, mô hình lệch số đo nhiều nhất 0,40 giây; tổng 100 mẫu 275 giây ở cả đo lẫn mô hình. Giải mã chiếm 73% thời gian agent. Lệnh: `python fable-work/out/LM-50/lm50_a30.py`.

**[ước từ mô hình đã khớp]** Áp mô hình vào số token đã đo của cả 1.686 mẫu ở mục 3 (161, 169, 98q), cộng `sec_asr` đã đo của từng mẫu:

| Máy nhà, tuần tự, 1.686 mẫu | p50 | p95 | p99 | Chậm nhất | Quá 20 giây |
|---|---:|---:|---:|---:|---:|
| Cả mẫu | 4,93 | 8,55 | 10,06 | 13,02 | 0 |

Trung bình mỗi mẫu: giải mã 2,09 giây, prefill và phần cố định 0,76, ASR chia đều 2,32. Mẫu chậm nhất là `train_0790` (8 lượt gọi): giải mã 9,4 + prefill 1,2 + ASR 2,4 giây.

### 4.4. Ước sang A30

Thông số công bố:

| GPU | FP32 | Băng thông bộ nhớ | Ghi chú |
|---|---:|---:|---|
| RTX 5060 Ti 16 GB (máy nhà) | 23,7 TFLOPS | 448 GB/s | không có số tensor core công bố |
| A30 24 GB | 10,3 TFLOPS | 933 GB/s | FP16 tensor core 165 TF |
| RTX 5090 32 GB | 104,8 TFLOPS | 1.792 GB/s | FAQ: BTC chạy lại Docker trên 2×5090 |

Nguồn: [datasheet A30](https://www.altair.com/pdfs/gpu-solutions/gpus/nvidia-a30-datasheet.pdf), [Tom's Hardware về RTX 5060 Ti 16GB](https://www.tomshardware.com/pc-components/gpus/nvidia-geforce-rtx-5060-ti-16gb-review), [TechPowerUp RTX 5060 Ti 16 GB](https://techpowerup.com/gpu-specs/geforce-rtx-5060-ti-16-gb.c4246), [Tom's Hardware về RTX 5090](https://www.tomshardware.com/pc-components/gpus/nvidia-geforce-rtx-5090-review).

**[ước]** Hệ số thời gian so với máy nhà, từ hai tỉ lệ công bố:

- A30: từ ×0,48 (448 / 933, khi việc bị giới hạn bởi băng thông bộ nhớ) tới ×2,30 (23,7 / 10,3, khi bị giới hạn bởi tính toán FP32 không qua tensor core).
- RTX 5090: ×0,23 tới ×0,25 ở cả hai tỉ lệ.

Giải mã ở batch 1 phải đọc toàn bộ 8,4 GB trọng số cho mỗi token, nên trần lý thuyết trên máy nhà là 448 / 8,4 ≈ 53 token/giây; số đo là 45, tức 85% trần. Vì vậy tôi coi giải mã là phần bị giới hạn bởi băng thông (×0,48 trên A30), còn prefill và ASR là phần bị giới hạn bởi tính toán, lấy hệ số xấu nhất ×2,30. Đây là suy luận từ một điểm đo, chưa có số đo nào trên A30.

| Cả mẫu, tuần tự, 1.686 mẫu | p50 | p95 | p99 | Chậm nhất | Quá 20 | Quá 25 (bị cắt) | Quá 30 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Máy nhà (mục 4.3) | 4,93 | 8,55 | 10,06 | 13,02 | 0 | 0 | 0 |
| A30, mọi phần ×0,48 | 2,37 | 4,10 | 4,83 | 6,25 | 0 | 0 | 0 |
| A30, giải mã ×0,48, prefill + ASR ×2,30 (tôi cho là sát nhất) | 7,28 | 13,33 | 18,49 | 20,32 | 1 | 0 | 0 |
| A30, mọi phần ×2,30 (xấu nhất) | 11,35 | 19,67 | 23,14 | 29,95 | 72 | 9 | 0 |
| RTX 5090, mọi phần ×0,25 | 1,2 | 2,1 | 2,5 | 3,3 | 0 | 0 | 0 |

Đọc bảng: ở giả định xấu nhất, 9 trên 1.686 mẫu (0,5%) chạm mức cắt 25 giây và nộp dở, không mẫu nào quá 30 giây. Ở giả định tôi cho là sát nhất không mẫu nào bị cắt. Hệ số lớn nhất mà mẫu chậm nhất vẫn không bị cắt là 25 / 13,02 = 1,92 cho mọi phần cùng lúc.

Biên an toàn trên máy nhà: mẫu chậm nhất cách mức cắt 12,0 giây và cách hạn 30 giây 17,0 giây.

Hai điều bảng này chưa phủ:

- `sec_asr` là phần chia đều theo lô. ASR N-best 4 đo riêng từng mẫu chưa có; trong M4 phần chia đều lớn nhất là 7,23 giây. Phải đo ở ngày kiểm (bước 2 mục 6).
- Private có app mới: prompt dài hơn và có thể nhiều lượt gọi hơn. Mô hình cho biết giá: mỗi 1.000 token prompt mới thêm 0,14 giây, mỗi 100 token ra thêm 2,2 giây trên máy nhà.

Nếu máy chấm thật sự chậm hơn máy nhà: giảm lượt nghe (ASR N-best batch 4 trên card 24 GB, ghi chú trong config 98n) trước khi đụng tới agent.

## 5. Chạy lại ra cùng kết quả

FAQ LifeMate (câu về |ΔPA| ≤ 0,03): BTC chạy lại tối đa 2 lần, trên 2×5090; chạy theo lô hay tuần tự do đội quyết định; BTC không sửa code. Tiêu chí là lệch PA không quá 0,03, không phải trùng từng byte. Brief gốc của LM-50 ghi "một A30"; hai nguồn khác nhau về máy, tôi tính cho cả hai ở mục 4.

Vì sao không dùng được cách của bản Qwen3: bản Qwen3 chạy `VLLM_BATCH_INVARIANT=1` (`DETERMINISTIC=1` trong `scripts/serve_vllm.sh`). Theo brief, vLLM chưa có kernel bất biến theo batch cho lớp Gated DeltaNet; tôi chưa tự kiểm vLLM báo gì khi bật biến này với Qwen3.5.

Chế độ lặp lại được **trên lý thuyết** (đã là mặc định của `configs/docker-qwen35.yaml` + `predict.sh`):

- `workers: 1` và `--max-num-seqs 1`: mỗi lúc chỉ một chuỗi, nên không có cách ghép batch nào phụ thuộc thời điểm request tới.
- `temperature: 0.0`, `seed: 0`, `--seed 0`: chọn token lớn nhất, không lấy mẫu.
- Mỗi lượt là container mới: prefix cache bắt đầu rỗng và lớn lên theo cùng thứ tự request.
- ASR chạy theo lô cố định trước khi vLLM lên, như bản 98 đã ra hai lượt trùng byte.

Điều lý thuyết không bảo đảm: `docs/BASELINE.md` ghi với Qwen3 không bật batch-invariant, chạy tuần tự vẫn chỉ 12/15 mẫu giống nhau. Tôi chưa biết nguyên nhân (thử nghiệm đó có thể dùng lại server giữa hai lượt). Vì vậy đây là việc phải đo, không được coi là đã đạt.

Chi phí thời gian của chế độ tuần tự **[ước từ số đo]**: Qwen3.5 tuần tự mất 275 giây pha agent cho 100 mẫu (2,75 giây mỗi mẫu, M5), tức khoảng 29 phút cho 634 mẫu; ở 8 luồng pha agent của 98q mất 643 giây cho 634 mẫu. Tuần tự chậm khoảng 2,7 lần ở pha agent, thêm khoảng 18 phút cho 634 mẫu. Cả lượt 634 mẫu ước 66 phút: ASR 33–34 phút **[đo, M3]**, khởi động vLLM 3 phút **[đo, M1: 181 giây]**, agent khoảng 29 phút **[ước từ M5]**.

Nếu tuần tự hoá ra không trùng byte, mức lệch có thể chấp nhận được theo tiêu chí của BTC. **[đo]** Hai lượt 161 ở 8 luồng, `DETERMINISTIC=0`, cùng model (`pro-threshold/runs` và `fable-eval/lm49-max-vote-repro-dbe2e04/runs`):

| Tập | Mẫu | Mẫu khác dự đoán | Mẫu đổi điểm | PA hai lượt | Lệch PA |
|---|---:|---:|---:|---|---:|
| val_app | 301 | 12 | 3 | 0,4628 và 0,4594 | 0,0033 |
| val_random | 225 | 4 | 0 | 0,6699 và 0,6699 | 0,0000 |

Chế độ kém lặp lại nhất mà đội đang dùng lệch khoảng một phần chín ngưỡng 0,03, trên cùng một GPU. Lệch giữa hai GPU khác kiến trúc chưa đo được ở đây.

## 6. Việc phải chạy thật vào ngày kiểm Docker

Theo thứ tự; thời gian GPU là ước lượng cho máy nhà, từ các số đo ở mục 4. Mọi việc xếp qua khoá GPU chung, dữ liệu là train (val_app), không dùng public_test.

| # | Việc | Lệnh | GPU |
|---|---|---|---:|
| 0 | Kiểm image bằng CPU | `scripts/docker_cpu_check.sh` | 0 |
| 1 | Thử 4 mẫu, một lượt | `fable-work/out/LM-50/b_check.sh tiny 4 val_app` | 4 phút (đã đo: 3 phút 13 giây) |
| 2 | Tuần tự, 2 lượt, 150 mẫu val_app, kèm đo ASR N-best từng mẫu: so từng byte, lấy p50 / p95 / p99 / chậm nhất | `b_check.sh seq 150 val_app` | 45 phút |
| 3 | Chỉ khi bước 2 không trùng byte: tắt prefix cache, rồi bật `VLLM_BATCH_INVARIANT`, mỗi biến thể 2 lượt 60 mẫu | `b_check2.sh seq-nocache 60 val_app`, `b_check2.sh seq-inv 60 val_app` | 36 phút |
| 4 | Tuỳ chọn: 8 luồng, 2 lượt, 150 mẫu, để biết mức lệch PA của chế độ nhanh | `b_check.sh par8 150 val_app` | 28 phút |
| 5 | Dựng image với model nộp, chạy lại bước 0 và 1 | `LLM=… TAG=… scripts/docker_build.sh` | 4 phút GPU, 5 phút CPU |
| 6 | Nếu chủ dự án muốn: hai lượt đủ cỡ 634 mẫu với image nộp | `scripts/docker_check.sh` | 2 giờ 45 phút |

Cách ước: mỗi lượt 150 mẫu gồm ASR 150 × 3,2 giây (M3), khởi động vLLM 3 phút (M1), agent 150 × 2,75 giây (M5), tức khoảng 18 phút; đo ASR từng mẫu thêm khoảng 8 phút. Đường ngắn nhất (0, 1, 2, 5) cần khoảng 55 phút GPU; đủ cả 3 và 4 cần khoảng 2 giờ.

Sau bước 2, điền vào mục 4 và 5: số mẫu khác nhau giữa hai lượt, phân bố độ trễ tuần tự trong container (so với M5), ASR N-best từng mẫu, số mẫu chạm mức cắt.

## 7. Hai ASR trên một PhoWhisper gốc (brief mục 0000, 12/11)

Công thức 195 / 98r3 dùng hai ASR: `phowhisper-large-ft` (beam 4, N-best 4) và bản chép greedy của `phowhisper-large-ft2-all` làm phương án thêm (`asr.merge_with`). Hai bản đã gộp cộng Qwen3.5 là 7.292.361.216 tham số, vượt 6 tỷ. Cả hai là LoRA hạng 32 trên `vinai/PhoWhisper-large`, nên image mang một bản gốc + hai adapter.

**[đo]** Đếm từ safetensors (`docker_report.py --params-only`):

| Model | Tham số |
|---|---:|
| `models/qwen35-4b-sft-full-para-v1` | 4.205.751.296 |
| `models/phowhisper-large-base` (fp32) | 1.543.304.960 |
| `models/phowhisper-large-ft-lora` | 57.671.680 |
| `models/phowhisper-large-ft2-all-lora` | 57.671.680 |
| **Tổng** | **5.864.399.616** (còn 135.600.384 tới 6 tỷ) |

Còn chỗ cho đúng hai adapter cỡ này nữa (115.343.360); adapter thứ ba cùng cỡ thì vượt. LM-52 đếm phương án bốn adapter của config 185 / 186 là 5.967.946.496, và nếu giữ thêm ft2-all thì 6.025.618.176, quá trần.

**Adapter.** Adapter gốc không còn, nên `scripts/whisper_lora_extract.py` tách lại từ (bản gộp fp16 − bản gốc fp32) bằng SVD hạng 32 trên 512 Linear đích, chỉ CPU. **[đo]** Phần ngoài hạng 32 nhỏ (σ₃₃ / σ₁ lớn nhất 0,0026 và 0,0013); không tensor nào ngoài lớp đích khác bản gốc. Gộp lại không trùng từng bit bản cũ: 12,74% phần tử lệch đúng một bậc làm tròn fp16 (lớn nhất 0,000244), vì bản gộp cũ đã làm tròn về fp16.

**Suy luận.** `hf_whisper` nhận `adapter: <thư mục>`: nạp bản gốc fp32, cộng adapter (`merge_lora_into`, không cần peft), rồi về fp16. Tốc độ như bản gộp; nạp model chậm hơn vì đọc 5,8 GB fp32. Hai ASR chạy lần lượt, không lúc nào có hai model trong bộ nhớ.

**Config và dựng** (chưa dựng, chưa chạy): `configs/docker-qwen35-asr2.yaml`; `Dockerfile` có `VARIANT=full2`; lệnh ở đầu file config.

Ba lưu ý (LM-52 nêu 12/11, áp dụng cả cho 98r3):

1. `asr.merge_with` trước đây chỉ giữ `sec` của ASR chính, nên thời gian chép của ASR thứ hai không vào `time_budget` và không vào bảng độ trễ ở mục 4. Nhánh này đã sửa: `sec` của mẫu là tổng hai lượt nghe. Việc này không đổi kết quả của các lượt đo val (không có `time_budget`), chỉ làm `sec_p*` trong `metrics.json` lớn hơn. Bảng mục 4 chưa gồm lượt nghe thứ hai; số duy nhất đang có là ASR greedy đo riêng từng mẫu của bản 98: p50 0,6, p99 1,8, chậm nhất 2,2 giây.
2. ASR thứ hai chép cả các lượt lịch sử dù chỉ dùng lượt cuối. Config Docker đặt `history: false` cho nó; khoá cache đổi, nhưng trong Docker cache luôn mới.
3. Thêm ASR thứ ba, thứ tư (185 / 186) thì cả hai mục trên lớn theo, và cần tách thêm LoRA của `ft2`.

**[đo] Bản chép của gốc + adapter so với bản gộp** (`scripts/check_asr_adapter.py`, 50 mẫu val_app, 211 lượt nói, máy nhà, 12/11 12:11):

| ASR | Cách chép | Lượt nói khác bản chép chính | Lượt nói khác ở bất kỳ phương án nào | Mẫu bị ảnh hưởng | Thời gian (gộp / gốc + adapter) |
|---|---|---:|---:|---:|---|
| `phowhisper-large-ft` | beam 4, N-best 4 | 3 / 211 | 8 / 211 | 8 / 50 (5 mẫu ở lượt cuối) | 143,4 / 142,0 giây |
| `phowhisper-large-ft2-all` | greedy | 1 / 211 | 1 / 211 | 1 / 50 | 31,5 / 31,2 giây |

Khác nhau là do một bậc làm tròn fp16 ở 12,7% trọng số; beam search nhạy hơn greedy. Tốc độ không đổi. Hệ quả: bản nộp một gốc + hai adapter **không cho đúng từng bản chép** của các lượt 195 / 98r3 đã đo bằng bản gộp; ảnh hưởng tới điểm chưa đo (cần một lượt val 195 chạy bằng config adapter). Cùng phép đo cho biết giá của lượt nghe thứ hai: 31,5 giây cho 50 mẫu, tức 0,63 giây mỗi mẫu khi chép cả lịch sử, so với 2,87 giây mỗi mẫu của lượt N-best 4.
