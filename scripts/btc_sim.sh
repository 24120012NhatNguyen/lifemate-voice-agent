#!/usr/bin/env bash
# Simulate the organisers' (BTC) check of a submitted image, step by step as in their Docker guide:
#   md5sum → docker load → docker run --gpus '"device=0"' -v <data>:/data:ro -v <out>:/result IMAGE /bin/bash /code/predict.sh
#   (plus --network none: grading has no network) → check /result/submission.jsonl → run predict_notebook.ipynb headless
#   (as "Run all") → /result/jupyter_submission.jsonl must equal submission.jsonl apart from `time` → optional comparison
#   with a reference predictions file → start_jupyter.sh answers on port 9777 with token zac2026.
#
#   flock "$GPU_LOCK" scripts/btc_sim.sh artifacts/submission/<name>.tar.gz [data_dir] [reference.jsonl]
#
# data_dir: what BTC mounts as /data (contains private_test/), or a split folder itself (mounted as /data/<split>, so
#   only that split is visible, like at grading). Default: ../data/lifemate/public_test.
# reference.jsonl: predictions to compare tool_calls/state_hash with (e.g. fable-eval/v_config270.jsonl).
# Env: OUT (default artifacts/btc_sim/<time>), LIMIT (smoke test on N samples; passed to the container with -e, which
#   BTC does not do), SKIP_LOAD=1 IMAGE=<tag> (image already loaded), NOTEBOOK=0 (skip the notebook run),
#   JUPYTER_CHECK=0 (skip the port-9777 check).
# Exit 0 only if every check passes.
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
TAR=${1:?usage: btc_sim.sh <image.tar.gz> [data_dir] [reference.jsonl]}
DATA=$(readlink -f "${2:-$HERE/../data/lifemate/public_test}")
REF=${3:-}
OUT=$(readlink -f "${OUT:-$HERE/artifacts/btc_sim/$(date +%Y%m%d-%H%M%S)}")
mkdir -p "$OUT/result"
LOG=$OUT/btc_sim.log
fail=0
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
ok() { say "PASS  $*"; }
bad() { say "FAIL  $*"; fail=1; }

say "image file $TAR, data $DATA, out $OUT${LIMIT:+, LIMIT $LIMIT}${REF:+, reference $REF}"
if [ "${SKIP_LOAD:-0}" != 1 ]; then
  md5=$(md5sum "$TAR" | cut -d' ' -f1)
  say "md5 $md5  ($(du -h "$TAR" | cut -f1))"
  if [ -f "$TAR.md5" ]; then
    [ "$(cut -d' ' -f1 "$TAR.md5")" = "$md5" ] && ok "md5 matches $TAR.md5" || bad "md5 differs from $TAR.md5"
  fi
  t=$(date +%s)
  loaded=$(docker load -i "$TAR" | tee -a "$LOG" | sed -n 's/^Loaded image: //p' | tail -1)
  [ -n "$loaded" ] && ok "docker load: $loaded ($(( $(date +%s) - t )) s)" || { bad "docker load printed no image"; exit 1; }
  IMAGE=${IMAGE:-$loaded}
fi
IMAGE=${IMAGE:?set IMAGE when SKIP_LOAD=1}
docker image inspect "$IMAGE" --format '{{.Id}} {{.Size}} {{json .Config}}' | python3 -c '
import json, sys
i, size, cfg = sys.stdin.read().split(" ", 2)
c = json.loads(cfg)
print("image", i, "size %.1f GB" % (int(size) / 1e9), "entrypoint", c.get("Entrypoint"), "cmd", c.get("Cmd"),
      "workdir", c.get("WorkingDir"))' | tee -a "$LOG"

if [ -f "$DATA/$(basename "$DATA").jsonl" ]; then
  split=$(basename "$DATA"); MOUNT=(-v "$DATA:/data/$split:ro")
else
  split=$(cd "$DATA" && for d in */; do d=${d%/}; [ "$d" != train ] && [ -f "$d/$d.jsonl" ] && echo "$d"; done | grep -x private_test || \
          (cd "$DATA" && for d in */; do d=${d%/}; [ "$d" != train ] && [ -f "$d/$d.jsonl" ] && echo "$d"; done | head -1))
  MOUNT=(-v "$DATA:/data:ro")
fi
INPUT=$( [ -f "$DATA/$split.jsonl" ] && echo "$DATA/$split.jsonl" || echo "$DATA/$split/$split.jsonl")
EXTRA=()
[ -n "${LIMIT:-}" ] && EXTRA+=(-e "LIMIT=$LIMIT")
say "split $split, input $INPUT ($(wc -l < "$INPUT") lines)"

# ---- 1. BTC's command
say "docker run --gpus '\"device=0\"' --network none ${MOUNT[*]} -v $OUT/result:/result ${EXTRA[*]} $IMAGE /bin/bash /code/predict.sh"
t=$(date +%s)
docker run --rm --gpus '"device=0"' --network none "${MOUNT[@]}" -v "$OUT/result:/result" "${EXTRA[@]}" \
  "$IMAGE" /bin/bash /code/predict.sh > "$OUT/predict_stdout.log" 2>&1
rc=$?
say "predict.sh exit $rc, wall $(( $(date +%s) - t )) s; stdout in $OUT/predict_stdout.log"
[ $rc = 0 ] && ok "predict.sh exit code 0" || bad "predict.sh exit code $rc"
grep -E "load model:|predict: " "$OUT/predict_stdout.log" | tee -a "$LOG"
grep -q "load model:" "$OUT/predict_stdout.log" && grep -q "predict: " "$OUT/predict_stdout.log" \
  && ok "stdout reports model-load and predict timings" || bad "stdout lacks load/predict timings"
grep -qE '^\[[0-9-]+ [0-9:]+\] (ERROR: |WARNING: )' "$OUT/predict_stdout.log" && bad "predict.py reported an error or warning (see stdout)"

check_sub() {  # check_sub FILE: format check against the input
  python3 - "$1" "$INPUT" "${LIMIT:-}" <<'EOF'
import json, random, sys
path, inp, limit = sys.argv[1], sys.argv[2], sys.argv[3]
ids = [json.loads(x)["id"] for x in open(inp, encoding="utf-8") if x.strip()]
if limit and int(limit) < len(ids):
    ids = sorted(random.Random(2026).sample(ids, int(limit)))
rows = [json.loads(x) for x in open(path, encoding="utf-8")]
errs = []
if [r.get("id") for r in rows] != ids:
    errs.append(f"ids/order differ from input: {len(rows)} rows vs {len(ids)} expected")
for r in rows:
    if set(r) != {"id", "tool_calls", "state_hash", "time"}:
        errs.append(f"{r.get('id')}: keys {sorted(r)}")
    elif not isinstance(r["tool_calls"], list) or not all(isinstance(c, dict) and set(c) == {"name", "arguments"} for c in r["tool_calls"]):
        errs.append(f"{r['id']}: bad tool_calls")
    elif not (isinstance(r["state_hash"], str) and len(r["state_hash"]) == 64):
        errs.append(f"{r['id']}: bad state_hash")
    elif not (isinstance(r["time"], int) and r["time"] > 0):
        errs.append(f"{r['id']}: bad time {r['time']}")
t = sorted(r.get("time", 0) for r in rows) or [0]
print(f"{path}: {len(rows)} rows, time ms p50 {t[len(t)//2]} p95 {t[int(.95*(len(t)-1))]} max {t[-1]}, "
      f"over 30 s: {sum(x > 30000 for x in t)}, empty tool_calls: {sum(not r.get('tool_calls') for r in rows)}")
for e in errs[:10]:
    print("  ", e)
sys.exit(1 if errs else 0)
EOF
}
SUB=$OUT/result/submission.jsonl
if [ -s "$SUB" ]; then
  check_sub "$SUB" | tee -a "$LOG"; [ "${PIPESTATUS[0]}" = 0 ] && ok "submission.jsonl format, line count and ids" || bad "submission.jsonl check"
else
  bad "no /result/submission.jsonl"
fi

# ---- 2. notebook, "Run all" headless, fresh container like BTC's jupyter run
if [ "${NOTEBOOK:-1}" = 1 ]; then
  say "notebook: jupyter nbconvert --execute predict_notebook.ipynb (fresh container, --network none)"
  t=$(date +%s)
  docker run --rm --gpus '"device=0"' --network none "${MOUNT[@]}" -v "$OUT/result:/result" "${EXTRA[@]}" "$IMAGE" \
    /bin/bash -c 'cd /code && jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
      --output-dir /result --output jupyter_executed predict_notebook.ipynb' > "$OUT/notebook_stdout.log" 2>&1
  rc=$?
  say "notebook exit $rc, wall $(( $(date +%s) - t )) s; executed copy $OUT/result/jupyter_executed.ipynb"
  [ $rc = 0 ] && ok "notebook ran every cell without error" || { bad "notebook failed"; tail -20 "$OUT/notebook_stdout.log" | tee -a "$LOG"; }
  JSUB=$OUT/result/jupyter_submission.jsonl
  if [ -s "$JSUB" ]; then
    check_sub "$JSUB" | tee -a "$LOG"; [ "${PIPESTATUS[0]}" = 0 ] && ok "jupyter_submission.jsonl format" || bad "jupyter_submission.jsonl check"
    python3 - "$SUB" "$JSUB" <<'EOF' | tee -a "$LOG"
import json, sys
a = [json.loads(x) for x in open(sys.argv[1], encoding="utf-8")]
b = [json.loads(x) for x in open(sys.argv[2], encoding="utf-8")]
strip = lambda rows: [(r["id"], json.dumps(r["tool_calls"], sort_keys=True, ensure_ascii=False), r["state_hash"]) for r in rows]
diff = [x[0] for x, y in zip(strip(a), strip(b)) if x != y]
same = len(a) == len(b) and not diff
print(("IDENTICAL" if same else f"DIFFERENT on {len(diff)} of {len(a)}: {diff[:20]}") + " (jupyter_submission vs submission, time ignored)")
sys.exit(0 if same else 1)
EOF
    [ "${PIPESTATUS[0]}" = 0 ] && ok "jupyter_submission == submission (time ignored)" || bad "jupyter_submission differs from submission"
  else
    bad "no /result/jupyter_submission.jsonl"
  fi
fi

# ---- 3. reference predictions
if [ -n "$REF" ] && [ -s "$SUB" ]; then
  python3 - "$SUB" "$REF" <<'EOF' | tee -a "$LOG"
import json, sys
sub = {r["id"]: r for r in map(json.loads, open(sys.argv[1], encoding="utf-8"))}
ref = {r["id"]: r for r in map(json.loads, open(sys.argv[2], encoding="utf-8"))}
ids = [i for i in sub if i in ref]
key = lambda r: (json.dumps(r["tool_calls"], sort_keys=True, ensure_ascii=False), r["state_hash"])
diff = [i for i in ids if key(sub[i]) != key(ref[i])]
hash_diff = [i for i in ids if sub[i]["state_hash"] != ref[i]["state_hash"]]
print(f"vs reference {sys.argv[2]}: {len(ids)} common ids, identical {len(ids) - len(diff)}, "
      f"different {len(diff)} (state_hash different {len(hash_diff)}): {diff[:30]}")
EOF
fi

# ---- 4. start_jupyter.sh answers with token zac2026 (no GPU needed)
if [ "${JUPYTER_CHECK:-1}" = 1 ]; then
  name=btc-sim-jupyter-$$
  docker run -d --rm --name "$name" -p 127.0.0.1:19777:9777 "${MOUNT[@]}" -v "$OUT/result:/result" "$IMAGE" \
    /bin/bash /code/start_jupyter.sh > /dev/null
  code_tok=000; code_no=000
  for _ in $(seq 60); do
    code_tok=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:19777/api/contents/predict_notebook.ipynb?token=zac2026")
    [ "$code_tok" = 200 ] && break; sleep 1
  done
  code_no=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:19777/api/contents/predict_notebook.ipynb")
  docker logs "$name" > "$OUT/jupyter_server.log" 2>&1
  docker stop "$name" > /dev/null 2>&1
  [ "$code_tok" = 200 ] && [ "$code_no" = 403 ] && ok "start_jupyter.sh: port 9777, token zac2026 accepted (200), no token refused (403)" \
    || bad "start_jupyter.sh: with token $code_tok, without $code_no (see $OUT/jupyter_server.log)"
fi

[ $fail = 0 ] && say "BTC SIMULATION: ALL PASS" || say "BTC SIMULATION: SOME CHECKS FAILED"
exit $fail
