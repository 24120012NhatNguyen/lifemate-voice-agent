"""ASR có lỗi nhân tạo, chỉ để dựng dữ liệu SFT chịu lỗi nghe (`scripts/build_sft.py --asr-config`), không dùng khi nộp.

ASR fine-tune đã học chính audio của train_rest nên chép lại nó gần như không sai (PhoWhisper-large-ft: WER 2,0 trên
train_rest, 9,3 trên val_random), và bản chép đó không dạy được agent cách xử lý khi nghe sai. Thêm nhiễu trắng vào
audio trước khi nhận dạng kéo WER về mức của audio chưa nghe, mà vẫn giữ kiểu chữ của chính mô hình (chữ hoa, dấu câu,
chữ số) như lúc chạy thật. Dùng ASR gốc chưa fine-tune thì nhiều lỗi hơn nhưng ra chữ thường, không dấu câu.
"""
from __future__ import annotations

import zlib

import numpy as np

from ..registry import register
from .adapters import HFWhisper
from .base import Result, load_audio


@register("asr", "hf_whisper_noisy", status="needs_deps")
class NoisyHFWhisper(HFWhisper):
    """HFWhisper trên audio thêm nhiễu trắng ở SNR cố định; nhiễu sinh theo đường dẫn file nên chạy lại ra cùng bản chép."""

    def __init__(self, snr_db=10.0, seed=0, **kw):
        super().__init__(**kw)
        self.snr_db = snr_db
        self.seed = seed

    def _noisy(self, path) -> np.ndarray:
        wav = load_audio(path)
        rng = np.random.default_rng([self.seed, zlib.crc32(str(path).encode())])
        power = max(float(np.mean(wav ** 2)), 1e-10)
        return wav + rng.normal(0.0, np.sqrt(power / 10 ** (self.snr_db / 10)), wav.shape).astype(np.float32)

    def transcribe(self, reqs):
        out: list[Result] = []
        for i in range(0, len(reqs), self.batch_size):
            out += [Result(t, []) for t in self._gen([self._noisy(r.path) for r in reqs[i:i + self.batch_size]])]
        return out
