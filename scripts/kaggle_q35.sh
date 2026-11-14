#!/usr/bin/env bash
# Qwen3.5-4B (chỉ văn bản) trên Kaggle T4 ×2: dựng gói, đẩy, xem trạng thái, tải adapter về gộp, chạy tiếp phiên sau.
# Chạy trên PC của Pro trong một worktree có nhánh này (LM-50). Mọi việc chạm token Kaggle đi qua script của Pro
# (tools-pro/kpush.sh, kaggle_fetch.sh, kstat trong night_lib.sh): file này không đọc ~/.kaggle và không in token.
#
#   scripts/kaggle_q35.sh push   ACCT USER TÊN <file>.jsonl ["đối số thêm cho run_sft.py"]   một thí nghiệm, 2 GPU
#   scripts/kaggle_q35.sh smoke  ACCT USER TÊN <file>.jsonl     4 bước: dừng sau bước 2, chạy tiếp, nén adapter (~20 phút)
#   scripts/kaggle_q35.sh status ACCT USER TÊN
#   scripts/kaggle_q35.sh fetch  ACCT USER TÊN                  tải zip + log; có adapter thì gộp và gắn vào models/
#   scripts/kaggle_q35.sh resume ACCT USER TÊN <file>.jsonl ["đối số thêm"]   phiên sau, từ checkpoint mà fetch đã tải
#   scripts/kaggle_q35.sh pack   TÊN <file>.jsonl THƯ_MỤC       chỉ dựng gói vào THƯ_MỤC (phải chưa có)
#   scripts/kaggle_q35.sh auto   ACCT USER TÊN <file>.jsonl [TIỀN_TỐ_CONFIG] ["đối số thêm"]
#       một lệnh cho cả thí nghiệm: đẩy, chờ, phiên nào hết hạn thì tự chạy tiếp, gộp, rồi đo
#       TIỀN_TỐ_CONFIG{_val-app,}.yaml qua khoá GPU chung; ghi mốc vào night.log. Chạy tách rời:
#         setsid nohup scripts/kaggle_q35.sh auto nbinhdz nbinhdz <tên> <file>.jsonl configs/experiments/<tiền tố> &
#       SKIP_PUSH=1: kernel đã đẩy rồi, chỉ theo tiếp.
#
# ACCT là tên file token (nbinh, hisu, nbinhdz), USER là tên người dùng Kaggle (nhatnguyenhcmusk24, hiuinhc, nbinhdz).
# Kernel là USER/lifemate-q35-TÊN, dataset riêng tư là USER/lifemate-q35-TÊN-data. Đối số thêm đi sau "--" tới
# train_sft.py, ví dụ: push ... "--deadline 300 -- --epochs 2 --seed 1". Mặc định --deadline 640 phút.
# Xong: models/qwen35-4b-sft-TÊN (đã gộp) trong worktree pro-threshold; đo bằng tools-pro/gpu_eval.sh như 161.
# Biến: TOOLS_PRO, PT (worktree pro-threshold), KAGGLE_OUT, MODELS_DIR, Q35_DEPS (thư mục có peft cho .venv-vllm),
# DEADLINE (phút, mặc định 640), NO_MERGE=1 (fetch chỉ tải và giải nén, không gộp: dùng cho adapter của smoke).
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
ROOT=/path/to/workspace
TOOLS_PRO=${TOOLS_PRO:-$ROOT/fable-eval/tools-pro}
PT=${PT:-$ROOT/fable-eval/pro-threshold}
KAGGLE_OUT=${KAGGLE_OUT:-$ROOT/fable-eval/kaggle}      # nơi kaggle_fetch.sh ghi (cố định trong script đó)
MODELS_DIR=${MODELS_DIR:-$ROOT/fable-eval/models}
Q35_DEPS=${Q35_DEPS:-$ROOT/relay-codex/work/qwen35/.test-deps}
DEADLINE=${DEADLINE:-640}

usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 2; }
die() { echo "kaggle_q35: $*" >&2; exit 1; }
check_name() { [[ "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "tên thí nghiệm không hợp lệ: $1"; }

pack() {  # pack TÊN DATA DIR: thư mục phẳng đúng như run_sft_q35.py mô tả
  local name=$1 data=$2 dir=$3
  check_name "$name"
  [ -f "$data" ] || die "không thấy dữ liệu $data"
  [ ! -e "$dir" ] || die "$dir đã có: chọn thư mục mới"
  mkdir -p "$dir"
  cp "$HERE/kaggle/run_sft_q35.py" "$dir/run_sft.py"
  cp "$HERE/scripts/train_sft_q35r.py" "$dir/train_sft.py"
  cp "$HERE/scripts/export_text_only.py" "$HERE/prompts/qwen35_scaffold.jinja" "$dir/"
  cp "$data" "$dir/"
  (cd "$HERE" && git rev-parse HEAD 2>/dev/null; git -C "$HERE" status --porcelain 2>/dev/null | head -5) > "$dir/PACK_VERSION.txt" || true
}

push() {  # push ACCT USER TÊN DATA "RUN_ARGS" [file checkpoint phẳng...]
  local acct=$1 user=$2 name=$3 data=$4 run_args=$5; shift 5
  local slug=lifemate-q35-$name stage
  stage=$(mktemp -d "${TMPDIR:-/tmp}/kaggle_q35.XXXXXX")
  pack "$name" "$data" "$stage/pack"
  local files=("$stage"/pack/*)
  if [ $# -gt 0 ]; then files+=("$@"); fi
  echo "đẩy $user/$slug: run_sft.py $run_args (${#files[@]} file, $(du -shc "${files[@]}" | tail -1 | cut -f1))"
  "$TOOLS_PRO/kpush.sh" "$acct" "$user" "$slug" "$run_args" -- "${files[@]}"
  [ -n "$stage" ] && [ -d "$stage/pack" ] && rm -rf -- "$stage"
  echo "trạng thái: scripts/kaggle_q35.sh status $acct $user $name ; xong thì: scripts/kaggle_q35.sh fetch $acct $user $name"
}

run_args() {  # run_args TÊN DATA "ĐỐI SỐ THÊM": thêm --ddp, TÊN=FILE và --deadline nếu chưa có
  local name=$1 data=$2 extra=${3:-} head tail=""
  head=$extra
  if [[ " $extra " == *" -- "* ]]; then head=${extra%% -- *}; tail=" -- ${extra#* -- }"; fi
  if [[ "$extra" == "-- "* ]]; then head=""; tail=" $extra"; fi
  [[ " $head " == *" --deadline "* ]] || head="--deadline $DEADLINE $head"
  echo "--ddp $head $name=$(basename "$data")$tail" | tr -s ' '
}

cmd=${1:-}; shift || true
case "$cmd" in
  pack)
    [ $# -eq 3 ] || usage
    pack "$1" "$2" "$3"; ls -la "$3"
    ;;
  push|smoke)
    [ $# -ge 4 ] || usage
    acct=$1 user=$2 name=$3 data=$4 extra=${5:-}
    [ "$cmd" = smoke ] && extra="--smoke $extra"
    push "$acct" "$user" "$name" "$data" "$(run_args "$name" "$data" "$extra")"
    ;;
  resume)
    [ $# -ge 4 ] || usage
    acct=$1 user=$2 name=$3 data=$4 extra=${5:-}
    flat=$KAGGLE_OUT/lifemate-q35-$name/ckpt-flat
    ls "$flat"/ckpt__"$name"__state.pt >/dev/null 2>&1 || die "chưa có checkpoint ở $flat: chạy fetch trước"
    echo "chạy tiếp từ checkpoint: $(grep -o '"step": [0-9]*' "$flat/ckpt__${name}__meta.json" | head -1), $(du -sh "$flat" | cut -f1)"
    # --require-step: kernel phải thấy đúng checkpoint vừa đẩy (bước này trở lên). kpush đẩy kernel khi dataset còn ở bản
    # cũ thì bản cũ không có checkpoint (phiên 2) hoặc có checkpoint cũ hơn (phiên 3 trở đi): runner dừng sau khoảng một
    # phút thay vì học lại, và `auto` đẩy lại; lần đẩy lại thường gặp bản dataset của lần trước, đã có đúng checkpoint.
    step=$(grep -o '"step": [0-9]*' "$flat/ckpt__${name}__meta.json" | head -1 | grep -o '[0-9]*$')
    [ -n "$step" ] || die "không đọc được bước trong $flat/ckpt__${name}__meta.json"
    push "$acct" "$user" "$name" "$data" "$(run_args "$name" "$data" "--require-checkpoint --require-step $step $extra")" "$flat"/ckpt__"$name"__*
    ;;
  status)
    [ $# -eq 3 ] || usage
    # shellcheck disable=SC1091
    source "$TOOLS_PRO/night_lib.sh"
    s=$(kstat "$1" "$2/lifemate-q35-$3" || true)
    echo "${s:-không đọc được trạng thái}"
    ;;
  fetch)
    [ $# -eq 3 ] || usage
    acct=$1 user=$2 name=$3
    check_name "$name"
    slug=lifemate-q35-$name
    out=$KAGGLE_OUT/$slug
    "$TOOLS_PRO/kaggle_fetch.sh" "$acct" "$user/$slug"      # không kèm tên: chỉ tải các file zip và log ở gốc output
    [ -f "$out/q35_summary.log" ] && python3 - "$out/q35_summary.log" "$name" <<'PY' || true
import json, sys
s = json.load(open(sys.argv[1], encoding="utf-8"))
e = s.get("experiments", {}).get(sys.argv[2], {})
v = s.get("versions", {})
print(f"phiên {s.get('minutes')} phút, mã thoát {s.get('exit_code')}; torch {v.get('torch')}, transformers {v.get('transformers')}, "
      f"fla {v.get('flash-linear-attention')}, GPU {v.get('gpus')}")
print("các bước (giây):", s.get("stages"))
print({k: e.get(k) for k in ("status", "step", "total", "tok_s", "loss_first", "loss_last", "peak_alloc_gb_rank0",
                             "gpu_mem_peak_mib", "dropped_nonfinite", "fallback_gdn", "fallback_conv", "device_line")})
PY
    zip=$out/qwen35-4b-sft-$name-lora.zip
    ckpt=$out/qwen35-4b-sft-$name-ckpt.zip
    if [ -f "$zip" ]; then
      adapter=$out/qwen35-4b-sft-$name-lora
      [ -f "$adapter/adapter_model.safetensors" ] || (cd "$out" && python3 -m zipfile -e "$zip" .)
      if [ -n "${NO_MERGE:-}" ]; then echo "đã tải và giải nén (NO_MERGE): $adapter"; exit 0; fi  # vd. adapter 4 bước của smoke
      merged=$MODELS_DIR/qwen35-4b-sft-$name
      if [ ! -f "$merged/config.json" ]; then
        echo "gộp LoRA trên CPU (khoảng 3 phút, 17 GB RAM): $merged"
        (cd "$HERE" && CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=$Q35_DEPS \
          "$PT/.venv-vllm/bin/python" scripts/merge_lora.py "$adapter" "$merged" --base "$PT/models/qwen35-4b-text") \
          > "$out/merge-$name.log" 2>&1 || { echo "gộp lỗi, xem $out/merge-$name.log"; tail -5 "$out/merge-$name.log"; exit 1; }
      fi
      ln -sfn "$merged" "$PT/models/qwen35-4b-sft-$name"
      echo "sẵn sàng: models/qwen35-4b-sft-$name (trong $PT)"
    elif [ -f "$ckpt" ]; then
      flat=$out/ckpt-flat
      mkdir -p "$flat"
      (cd "$flat" && python3 -m zipfile -e "$ckpt" .)
      echo "CHƯA XONG: checkpoint $(grep -o '"step": [0-9]*' "$flat/ckpt__${name}__meta.json" | head -1) ở $flat"
      echo "chạy tiếp: scripts/kaggle_q35.sh resume $acct $user $name <file>.jsonl"
    else
      echo "không có adapter lẫn checkpoint trong $out: xem $out/$slug.log và $out/$name.log"
      ls -la "$out" | tail -20
      exit 1
    fi
    ;;
  auto)
    [ $# -ge 4 ] || usage
    acct=$1 user=$2 name=$3 data=$4 cfg=${5:-} extra=${6:-}
    check_name "$name"
    # shellcheck disable=SC1091
    source "$TOOLS_PRO/night_lib.sh"   # say, kstat (đặt lại T, PT, LOG về đúng thư mục của Pro)
    kernel=$user/lifemate-q35-$name
    if [ -n "$cfg" ]; then   # thiếu config thì biết ngay, không phải sau cả chục giờ train
      for c in "${cfg}_val-app.yaml" "${cfg}.yaml"; do
        [ -f "$PT/$c" ] || [ -f "$HERE/$c" ] || die "không thấy $c ở $PT lẫn $HERE"
      done
    fi
    pushed=""
    if [ -z "${SKIP_PUSH:-}" ]; then
      "$0" push "$acct" "$user" "$name" "$data" "$extra" > /dev/null || { say "Q35-Kaggle $name: đẩy lỗi"; exit 1; }
      say "Q35-Kaggle $name: đã đẩy $kernel ($(basename "$data")), tự theo tới khi có model"
      pushed=1
    fi
    ok=""
    for round in $(seq 1 12); do   # phiên thật + các lần đẩy lại khi kernel gặp bản dataset cũ
      s=""
      if [ -n "$pushed" ]; then   # vừa đẩy: chờ Kaggle nhận phiên mới (tối đa 10 phút), kẻo đọc nhầm trạng thái cuối của phiên trước
        for _ in $(seq 1 20); do
          s=$(kstat "$acct" "$kernel" || true)
          case "$s" in *RUNNING|*QUEUED) break ;; esac
          sleep 30
        done
      fi
      for _ in $(seq 1 300); do   # tối đa 15 giờ mỗi phiên
        s=$(kstat "$acct" "$kernel" || true)
        case "$s" in *COMPLETE|*ERROR|*CANCEL*) break ;; esac
        sleep 180
      done
      r=""
      for _ in 1 2 3; do   # tải về có thể lỗi mạng nhất thời: thử lại
        r=$("$0" fetch "$acct" "$user" "$name" 2>&1 | tail -6) || true
        case "$r" in *"sẵn sàng:"*|*"CHƯA XONG"*) break ;; esac
        sleep 90
      done
      case "$r" in
        *"sẵn sàng:"*) ok=1; break ;;
        *"CHƯA XONG"*)
          say "Q35-Kaggle $name: phiên $round dừng giữa chừng ($(echo "$r" | grep -o '"step": [0-9]*' | head -1)), đẩy phiên chạy tiếp"
          "$0" resume "$acct" "$user" "$name" "$data" "$extra" > /dev/null || { say "Q35-Kaggle $name: đẩy phiên chạy tiếp lỗi"; exit 1; }
          pushed=1
          ;;
        *) say "Q35-Kaggle $name: phiên $round kết thúc ($s) mà không có adapter lẫn checkpoint: $(echo "$r" | tail -2 | tr '\n' ' ' | cut -c1-200)"; exit 1 ;;
      esac
    done
    [ -n "$ok" ] || { say "Q35-Kaggle $name: quá 6 phiên vẫn chưa xong"; exit 1; }
    sum=$(python3 -c "
import json
s = json.load(open('$KAGGLE_OUT/lifemate-q35-$name/q35_summary.log', encoding='utf-8'))
e = s['experiments']['$name']
print(f\"bước {e.get('step')}/{e.get('total')}, loss {e.get('loss_last')}, {e.get('tok_s')} tok/s, bỏ {e.get('dropped_nonfinite')} mẫu, bộ nhớ GPU đỉnh {e.get('gpu_mem_peak_mib')} MiB\")" 2>/dev/null || true)
    say "Q35-Kaggle $name: đã gộp models/qwen35-4b-sft-$name ($sum)"
    [ -n "$cfg" ] || exit 0
    for c in "${cfg}_val-app.yaml" "${cfg}.yaml"; do
      if [ ! -f "$PT/$c" ]; then   # config đã có sẵn ở worktree đo của Pro thì dùng luôn; chưa có thì chép từ nhánh này
        [ -f "$HERE/$c" ] || { say "Q35-Kaggle $name: không thấy $c ở $PT lẫn $HERE, không đo"; exit 1; }
        cp "$HERE/$c" "$PT/$c"
      fi
    done
    VX="--gdn-prefill-backend triton --max-num-seqs 8 --max-num-batched-tokens 8192 --chat-template $PT/prompts/qwen35_scaffold.jinja"
    res=$(DETERMINISTIC=0 GPU_UTIL=0.80 VLLM_EXTRA="$VX" "$TOOLS_PRO/gpu_eval.sh" "models/qwen35-4b-sft-$name" \
      "Qwen3.5-4B SFT $name (Kaggle T4 x2)" "${cfg}_val-app.yaml" "${cfg}.yaml" | tail -1)
    say "Q35-Kaggle $name đo xong (val_app, val_random): $res"
    ;;
  *)
    usage
    ;;
esac
