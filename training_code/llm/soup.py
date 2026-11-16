"""Model soup: trung bình trọng số của nhiều bản Qwen3.5 SFT đã gộp LoRA (cùng model gốc), đọc từng tensor để ít RAM.
Dùng: soup.py OUT MODEL1 MODEL2 [...] [--w 0.5,0.5]   (thư mục model trong pro-threshold/models)

Viết bởi Pro (fable-eval/tools-pro/soup.py, 12/11), đưa vào repo nguyên văn cho gói nộp (LM-48); chỉ tách câu lệnh
ghép dấu chấm phẩy cho qua ruff, hành vi không đổi."""
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

torch.set_num_threads(4)
args = sys.argv[1:]
w = None
if "--w" in args:
    i = args.index("--w")
    w = [float(x) for x in args[i + 1].split(",")]
    del args[i:i + 2]
out, srcs = Path(args[0]), [Path(a) for a in args[1:]]
w = w or [1 / len(srcs)] * len(srcs)
assert len(w) == len(srcs) and abs(sum(w) - 1) < 1e-6
files = [safe_open(str(s / "model.safetensors"), "pt") for s in srcs]
keys = list(files[0].keys())
assert all(set(f.keys()) == set(keys) for f in files[1:]), "khác tập tensor"
res, meta = {}, files[0].metadata()
for k in keys:
    ts = [f.get_tensor(k) for f in files]
    if ts[0].is_floating_point():
        acc = sum(wi * t.float() for wi, t in zip(w, ts))
        res[k] = acc.to(ts[0].dtype)
    else:
        res[k] = ts[0]
out.mkdir(parents=True, exist_ok=True)
save_file(res, str(out / "model.safetensors"), metadata=meta)
for f in srcs[0].iterdir():
    if f.name != "model.safetensors" and f.is_file():
        shutil.copy(f, out / f.name)
(out / "soup.json").write_text(json.dumps({"sources": [str(s) for s in srcs], "weights": w}, ensure_ascii=False))
print("soup xong:", out, len(keys), "tensor")
