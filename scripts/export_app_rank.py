"""Ghi thứ hạng app (đếm số lần được cài trong train, ít trước) ra JSON để Docker mồi ASR đúng thứ tự lúc train.

Docker không mang dữ liệu train; đặt LIFEMATE_APP_RANK trỏ tới file này. Chỉ chứa ID app của train, không chứa mẫu.
Dùng: python scripts/export_app_rank.py models/<thư-mục-ASR>/app_rank.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lifemate.asr.base import _app_rank  # noqa: E402

out = Path(sys.argv[1])
rank = _app_rank()
out.write_text(json.dumps(rank, ensure_ascii=False, indent=0, sort_keys=True), encoding="utf-8")
print(f"{out}: {len(rank)} app")
