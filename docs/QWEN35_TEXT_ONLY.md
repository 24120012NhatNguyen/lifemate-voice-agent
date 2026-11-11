# Qwen3.5-4B text-only · LM-43

Chuẩn bị offline từ `e9693ff` để có thể SFT nếu kết quả sàng lọc config 160 được quản lý chấp nhận. Chưa có bằng chứng tăng điểm; không train, eval hay phục vụ GPU trong LM-43.

## Artifact và số tham số

`python scripts/export_text_only.py` đọc duy nhất snapshot HF cục bộ `Qwen/Qwen3.5-4B`, revision đầy đủ `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Không gọi Hub. Không ghi đè output tồn tại. Output mặc định `models/qwen35-4b-text`.

| Nhóm trong safetensors gốc | Tham số |
| --- | ---: |
| language_model, giữ lại | 4.205.751.296 |
| vision, bỏ | 333.514.240 |
| MTP, bỏ | 120.599.552 |
| Tổng gốc | 4.659.865.088 |
| Tổng bỏ | 454.113.792 |

426 tensor giữ lại, đổi tiền tố `model.language_model.` → `model.`. `lm_head` buộc chung embedding, chỉ đếm một lần. Config mới: `qwen3_5_text`, `Qwen3_5ForCausalLM`, `mtp_num_hidden_layers: 0`. Meta model xác minh toàn bộ tên/shape/số tham số trước ghi. Không chỉ sửa config rồi để vision/MTP trong weights.

`export_receipt.json` chứa revision, từng nhóm và SHA-256/byte của từng file nguồn/đích. File receipt không tự hash chính nó; bản sao trong `reports/lm43/` có thể đối chiếu với file ở model. `generation_config.json` đặt EOS gồm cả `<|endoftext|>` gốc và `<|im_end|>` chat, pad theo tokenizer gốc.

**Tokenizer phải là `Qwen3_5Tokenizer`.** Trong Transformers 5.20.2, AutoTokenizer của wrapper gốc chọn lớp này dù tokenizer_config gốc ghi Qwen2. Sau khi đổi model_type thành text-only, giữ nguyên metadata cũ làm nó chọn Qwen2 và thay regex xử lý dấu tổ hợp. Exporter đặt rõ lớp Qwen3.5; checker so sánh toàn bộ tokenizer backend và chat template với gốc, thêm các chuỗi Việt/số/ngày/Unicode. Không dùng cờ sửa regex Mistral.

Theo số ASR ~1,543B trong brief, tổng dự kiến ~5,749B < 6B; LLM < 5B. Đây không thay cho biên nhận tham số ASR/chứng nhận thể lệ của quản lý.

## Môi trường và kiểm CPU

Dùng môi trường mới theo `kaggle/requirements-qwen35.txt`; không nâng Transformers trong môi trường ASR hiện tại. Đã kiểm CPU với torch 2.14.0, Transformers 5.20.2, safetensors 0.8.0, PEFT 0.21.2, Accelerate 1.15.0. PEFT/Accelerate được lấy từ cache có sẵn, không cài/tải gói mới. `.test-deps` chỉ là symlink cục bộ cho lượt kiểm này, không phải artifact triển khai.

```bash
export CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2
# PY_QWEN35 là Python của môi trường riêng có các phiên bản trên.
"$PY_QWEN35" scripts/export_text_only.py
"$PY_QWEN35" scripts/check_qwen35_cpu.py
"$PY_QWEN35" scripts/train_sft.py --base models/qwen35-4b-text \
  --inspect-only --out reports/lm43 --local-files-only
"$PY_QWEN35" kaggle/run_qwen35_sft.py check
```

`check_qwen35_cpu.py` xác minh hash, nạp model thật trên CPU bf16, thiếu/thừa key, trọng số buộc chung, số tham số và forward hữu hạn trên câu giả "Xin chào". Test thu nhỏ có cả DeltaNet/full attention, backward nhận gradient ở cả hai nhánh và MLP, lưu/nạp/gộp adapter, logits trước/sau gộp, cả entry point `merge_lora.py`. Test mã hoá dùng tokenizer cache, không dùng dữ liệu BTC. Hai config 161 cũng được đối chiếu sâu với zs0p.

## LoRA và chat

248 module Linear: 24×5 projection DeltaNet (`in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, `out_proj`), 8×4 attention (`q_proj`, `k_proj`, `v_proj`, `o_proj`), 32×3 MLP (`gate_proj`, `up_proj`, `down_proj`). Danh sách đầy đủ trong `reports/lm43/lora_modules.json`; trainer cũng in và lưu danh sách trong `train_args.json`. Không gắn LoRA vào convolution/norm/A_log/dt_bias.

Trainer phát hiện `qwen3_5_text` và dùng native template với `enable_thinking=False`. Template vẫn có khung `<think>` rỗng trong prompt; khung này không có loss. Chỉ nội dung assistant và `<|im_end|>` được học. Hội thoại được mở rộng thành **một ví dụ cho mỗi bước assistant**, với nguyên lịch sử trước bước đó, vì template loại khung think của assistant cũ khi có user turn mới. Mọi prefix token phải khớp prompt suy luận; sai thì dừng. Dữ liệu có reasoning/thinking hoặc token chat đặc biệt bị từ chối. Nhánh Qwen3 cũ giữ encoder và cách nạp nhiều lượt cũ.

Hệ quả: số ví dụ/token và cách chia batch khác công thức zs0p; không gọi đây là ablation chỉ thay backbone khi train. Có thể tăng đáng kể thời gian vì lặp lại prompt. Cutoff mặc định trainer là 12288 như cũ, runner Kaggle dùng 4096 để thử VRAM; nó **bỏ nguyên ví dụ** dài hơn cutoff, không cắt nhãn. Phải review số bị bỏ và chọn cutoff trước train đầy đủ.

## Kaggle 2×T4

Notebook `kaggle/qwen35_text_only.ipynb` và runner `kaggle/run_qwen35_sft.py`. Notebook chưa có output GPU. `check` chạy CPU; `train` kiểm kiến trúc, dùng mount model/data cục bộ, không tự upload/download. Hai T4 chạy hai seed độc lập, mỗi GPU giữ đầy đủ model, không cộng VRAM. Lỗi child trả mã lỗi cho notebook; mỗi seed có log và adapter riêng, chưa gộp.

T4 sm75 không có bf16 native: chọn fp16 + GradScaler. **Chưa thử wheel torch 2.14 trên Kaggle, FLA, causal-conv1d, fp16 stability hay VRAM của model đầy đủ trên sm75.** Môi trường kiểm CPU dùng reference PyTorch cho cả Gated DeltaNet/convolution; pass CPU không chứng minh kernel GPU chạy. Runner mặc định từ chối GPU trước Ampere; `--allow-sm75-reference` chấp nhận rủi ro fallback chậm. Không tự cài FLA/causal-conv1d chưa kiểm phiên bản. Với phiên bản Transformers cũ hơn, đường fallback/API có thể khác.

Khi có job được cấp: chạy smoke tiny forward/backward/optimizer trên từng GPU (runner tự làm), rồi model đầy đủ `--max-steps 2`, kiểm loss hữu hạn, VRAM, tok/s và mẫu bị bỏ. Sau đó dùng output mới và bỏ max-steps nếu ngân sách cho phép. Hai bước smoke **không phải** checkpoint SFT hoàn chỉnh. Không chuyển dữ liệu BTC sang Kaggle trước khi quản lý xác nhận dữ liệu/mount được phép.

## Lệnh RTX 5060 Ti 16 GB · chỉ chạy khi được cấp GPU

Các lệnh dưới đây chưa chạy. Quản lý phải đặt đúng `GPU_LOCK` chung đang dùng; không dùng khoá riêng theo checkout. Dùng Python môi trường Qwen3.5 riêng, không dùng `.venv` Transformers 4.x qua `scripts/gpu_job.sh` hiện tại.

```bash
: "${GPU_LOCK:?Cần đường dẫn gpu.lock chung do quản lý đặt}"
: "${PY_QWEN35:?Cần Python môi trường Qwen3.5 riêng}"
# Pilot 2 bước, output riêng; --dtype bf16 chỉ cho RTX hỗ trợ native.
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 flock "$GPU_LOCK" "$PY_QWEN35" scripts/train_sft.py \
  --base models/qwen35-4b-text --data sft/train_synth.jsonl \
  --out models/qwen35-4b-pilot --dtype bf16 --max-len 4096 \
  --max-steps 2 --no-merge --local-files-only
# Full sau khi duyệt cutoff, data và ngân sách; không dùng lại thư mục pilot.
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 flock "$GPU_LOCK" "$PY_QWEN35" scripts/train_sft.py \
  --base models/qwen35-4b-text --data sft/train_synth.jsonl \
  --out models/qwen35-4b-sft-synth --dtype bf16 --max-len 4096 \
  --epochs 1 --seed 0 --no-merge --local-files-only
# Gộp CPU, chỉ lại base khi adapter đến từ Kaggle.
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 "$PY_QWEN35" scripts/merge_lora.py \
  models/qwen35-4b-sft-synth-lora models/qwen35-4b-sft-synth \
  --base models/qwen35-4b-text --local-files-only
```

Ước lượng kế hoạch, **chưa benchmark**: nếu đạt 100–400 token/s trên RTX thì 1 triệu token sau mở rộng cần 0,7–2,8 giờ; 4,3 triệu cần khoảng 3–12 giờ, cộng nạp/gộp. Ví dụ mở rộng có thể nhiều token hơn 4,3 triệu; reference DeltaNet có thể chậm hơn khoảng giả định này hoặc OOM. Lấy tổng token in bởi loader và tốc độ pilot để thay ước lượng: `giờ = tổng token / tok_per_sec / 3600`. Không tự xin/chạy job 12 giờ theo dự báo này.

## Phục vụ và tiêu chí chưa đạt

Config 161 kế thừa zs0p ở đúng hai subset; chỉ đổi model và `extra_body.chat_template_kwargs.enable_thinking=false`.

vLLM 0.33.0 trong `.venv-vllm` đã resolve được lớp native `vllm.model_executor.models.qwen3_5.Qwen3_5ForCausalLM` trên CPU, không khởi tạo CUDA. Tuy nhiên môi trường sandbox thấy `vllm._C` không tồn tại (`reports/lm43/vllm-binary.json`), import ghi cảnh báo thiếu extension/FlashInfer. **Chưa đạt tiêu chí engine nạp được.** Cần quản lý kiểm lại wheel CUDA của môi trường thật, rồi cấp lượt smoke GPU riêng. Không coi resolve registry là nạp weights bằng vLLM.

Sau khi sửa môi trường và có GPU, dùng khoá chung:

```bash
: "${GPU_LOCK:?Cần khoá GPU chung}"
: "${VLLM_BIN:?Cần vllm 0.33 có extension CUDA hoạt động}"
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 flock "$GPU_LOCK" "$VLLM_BIN" serve \
  models/qwen35-4b-sft-synth --port 8000 --max-model-len 4096 \
  --dtype bfloat16 --gpu-memory-utilization 0.85 --generation-config auto --seed 0
```

Client phải gửi `chat_template_kwargs: {enable_thinking: false}` như config 161, temperature 0. Trước train có thể smoke bản `models/qwen35-4b-text` bằng cùng lệnh, đổi đường dẫn. Chạy câu giả, kiểm đáp án/VRAM và log keys; không dùng public_test. Chưa xác nhận tất định batch, latency, tương thích JSON constrained decoding trên bản này.
