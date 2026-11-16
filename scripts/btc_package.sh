#!/usr/bin/env bash
# Save a built image as the file BTC asks for (docker save | gzip) and its md5, as in their guide (section 11 / 8):
#   scripts/btc_package.sh IMAGE [FILE_NAME]
#   scripts/btc_package.sh lifemate:btc270                          → zac2026_lifemate_fg3.tar.gz (default name)
#   scripts/btc_package.sh lifemate:btc270 zac2026_TeamName.tar.gz  → the final name, once the team name is known
# Default name zac2026_lifemate_<model>, <model> = the image's LLM_MODEL folder without "qwen35-4b-soup-".
# → artifacts/submission/<name>.tar.gz, <name>.tar.gz.md5 (md5sum format), <name>.txt (image id, git commit, sizes).
# pigz (parallel gzip, same .gz format) is used when installed. Needs about 22 GB free.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
IMAGE=${1:?usage: btc_package.sh IMAGE [FILE_NAME]}
if [ -n "${2:-}" ]; then
  NAME=${2%.tar.gz}
else
  llm=$(docker image inspect "$IMAGE" --format '{{range .Config.Env}}{{println .}}{{end}}' | sed -n 's/^LLM_MODEL=//p')
  model=$(basename "${llm:?image has no LLM_MODEL}"); model=${model#qwen35-4b-soup-}
  NAME=zac2026_lifemate_$model
fi
[[ "$NAME" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "bad file name: $NAME"; exit 2; }
OUT=$HERE/artifacts/submission
mkdir -p "$OUT"
F=$OUT/$NAME.tar.gz
[ -e "$F" ] && { echo "$F exists; remove it first"; exit 2; }
GZ=gzip; command -v pigz > /dev/null && GZ="pigz -p ${THREADS:-8}"
t=$(date +%s)
docker save "$IMAGE" | $GZ > "$F.part"
mv "$F.part" "$F"
(cd "$OUT" && md5sum "$NAME.tar.gz" > "$NAME.tar.gz.md5")
{
  echo "file $F"
  echo "md5 $(cut -d' ' -f1 "$F.md5")"
  echo "bytes $(stat -c %s "$F") ($(du -h "$F" | cut -f1))"
  docker image inspect "$IMAGE" --format 'image {{index .RepoTags 0}} id {{.Id}} size {{.Size}} created {{.Created}}'
  echo "git $(git -C "$HERE" rev-parse --short HEAD)$(git -C "$HERE" diff --quiet HEAD || echo ' +uncommitted')"
  echo "saved in $(( $(date +%s) - t )) s"
} | tee "$OUT/$NAME.txt"
