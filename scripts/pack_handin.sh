#!/usr/bin/env bash
# Chỉ đóng gói cục bộ; Pro chạy sau khi chọn dữ liệu/model cuối. Không gọi mạng/GPU.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
exec "${PYTHON:-python3}" "$HERE/scripts/pack_handin.py" "$@"
