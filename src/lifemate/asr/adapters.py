"""Các phương án ASR. Mỗi lớp một phương án, chọn bằng `asr.name` trong config.

| name             | Mô hình mặc định                          | Giấy phép                         | Tham số |
| gold             | audio_text của train (trần, không nghe)   |                                   | 0       |
| hf_whisper       | mọi checkpoint Whisper trên HF: openai/whisper-large-v3-turbo (MIT), vinai/PhoWhisper-large (BSD-3), Qualcomm-AI-Research/PhoASR-whisper-small (BSD-3-Clear + Qualcomm RAI) | | 0,24–1,55 tỷ |
| faster_whisper   | Whisper qua CTranslate2 (nhanh hơn)       | theo checkpoint                   |         |
| chunkformer      | khanhld/chunkformer-ctc-large-vie         | CC BY-NC 4.0: hỏi BTC trước khi nộp | 0,11 tỷ |
| dialect_phonetic | arXiv 2605.24451, chưa công bố mã         |                                   |         |

`model` nhận cả id trên HF lẫn thư mục checkpoint fine-tune trong repo (vd. `models/phowhisper-large-ft`,
xem scripts/finetune_*.py); đường dẫn tương đối tính từ gốc repo.
"""
from __future__ import annotations

from ..config import REPO
from ..registry import register
from .base import ASR, Request, Result, load_audio


def _local(model: str) -> str:
    """Đường dẫn checkpoint trong repo (models/...) thành tuyệt đối; id HF giữ nguyên."""
    path = REPO / model
    return str(path) if path.is_dir() else model


@register("asr", "gold")
class GoldASR(ASR):
    """Dùng audio_text gold của train: đo trần khi nghe đúng hoàn toàn."""

    def __init__(self, **_):
        pass

    def transcribe(self, reqs: list[Request]) -> list[Result]:
        out = []
        for r in reqs:
            if r.turn == "final":
                text = r.sample.get("audio_text")
            else:
                k = int(r.turn[1:])
                users = [t for t in r.sample.get("history", []) if t.get("role") == "user"]
                text = users[k - 1].get("audio_text") if k <= len(users) else None
            if text is None:
                raise ValueError(f"{r.key}: split này không có audio_text, chọn một ASR thật")
            out.append(Result(text))
        return out


@register("asr", "hf_whisper", status="needs_deps")
class HFWhisper(ASR):
    """Whisper bất kỳ trên HuggingFace qua transformers (Whisper-turbo, PhoWhisper, PhoASR-whisper-small)."""

    def __init__(self, model="openai/whisper-large-v3-turbo", num_beams=1, batch_size=16, dtype="float16",
                 device="cuda:0", n_best=1, tokens_per_sec=10, generate=None, **_):
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        self.torch = torch
        self.dtype = getattr(torch, dtype)
        self.device = device
        self.proc = AutoProcessor.from_pretrained(_local(model))
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(_local(model), torch_dtype=self.dtype).to(device).eval()
        self.num_beams = max(num_beams, n_best)
        self.n_best = n_best
        self.batch_size = batch_size
        self.tokens_per_sec = tokens_per_sec  # chặn vòng lặp ảo giác: số token tối đa theo độ dài audio
        self.generate = generate or {}  # truyền thẳng vào model.generate, vd. {no_repeat_ngram_size: 4}

    def _gen(self, wavs, prompt=""):
        torch = self.torch
        feats = self.proc(wavs, sampling_rate=16000, return_tensors="pt", padding="max_length",
                          return_attention_mask=True)
        longest = max(len(w) for w in wavs) / 16000
        kw = dict(language="vi", task="transcribe", num_beams=self.num_beams,
                  max_new_tokens=min(220, int(longest * self.tokens_per_sec) + 16),
                  num_return_sequences=self.n_best, **self.generate)
        if prompt:
            kw["prompt_ids"] = torch.tensor(self.proc.get_prompt_ids(prompt), device=self.device)
        with torch.inference_mode():
            ids = self.model.generate(feats.input_features.to(self.device, self.dtype),
                                      attention_mask=feats.attention_mask.to(self.device), **kw)
        texts = self.proc.batch_decode(ids, skip_special_tokens=True)
        if prompt:  # transformers giữ phần mồi ở đầu chuỗi giải mã ở một số phiên bản
            texts = [t[len(prompt):].strip() if t.strip().startswith(prompt) else t.strip() for t in texts]
        return [t.strip() for t in texts]

    def transcribe(self, reqs):
        out: list[Result] = []
        if any(r.hint for r in reqs):  # mồi khác nhau từng mẫu: chạy lẻ
            for r in reqs:
                texts = self._gen([load_audio(r.path)], r.hint)
                out.append(Result(texts[0], texts[1:]))
            return out
        for i in range(0, len(reqs), self.batch_size):
            part = reqs[i:i + self.batch_size]
            texts = self._gen([load_audio(r.path) for r in part])
            for j in range(len(part)):
                group = texts[j * self.n_best:(j + 1) * self.n_best]
                out.append(Result(group[0], group[1:]))
        return out

    def close(self):
        del self.model
        self.torch.cuda.empty_cache()


@register("asr", "faster_whisper", status="needs_deps")
class FasterWhisper(ASR):
    """Whisper qua CTranslate2: nhanh, hỗ trợ initial_prompt và hotwords."""

    def __init__(self, model="mobiuslabsgmbh/faster-whisper-large-v3-turbo", beam_size=1, compute_type="float16",
                 device="cuda", hotwords=False, **_):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model, device=device, compute_type=compute_type)
        self.beam_size = beam_size
        self.hotwords = hotwords

    def transcribe(self, reqs):
        out = []
        for r in reqs:
            kw = {"hotwords": r.hint} if (self.hotwords and r.hint) else {"initial_prompt": r.hint or None}
            segs, _ = self.model.transcribe(load_audio(r.path), language="vi", beam_size=self.beam_size,
                                            vad_filter=False, condition_on_previous_text=False, **kw)
            segs = list(segs)
            text = " ".join(s.text.strip() for s in segs).strip()
            conf = min((s.avg_logprob for s in segs), default=None)
            out.append(Result(text, conf=conf))
        return out

    def close(self):
        del self.model


@register("asr", "chunkformer", status="needs_deps")
class ChunkFormer(ASR):
    """ChunkFormer CTC (110 triệu tham số, giấy phép CC BY-NC 4.0)."""

    def __init__(self, model="khanhld/chunkformer-ctc-large-vie", chunk_size=64, left_context_size=128,
                 right_context_size=128, total_batch_duration=1800, **_):
        from chunkformer import ChunkFormerModel

        self.model = ChunkFormerModel.from_pretrained(_local(model))
        self.kw = dict(chunk_size=chunk_size, left_context_size=left_context_size,
                       right_context_size=right_context_size, total_batch_duration=total_batch_duration)

    def transcribe(self, reqs):
        texts = self.model.batch_decode(audio_paths=[str(r.path) for r in reqs], **self.kw)
        return [Result(t if isinstance(t, str) else str(t)) for t in texts]

    def close(self):
        import torch
        del self.model
        torch.cuda.empty_cache()


@register("asr", "dialect_phonetic", status="research")
class DialectPhonetic(ASR):
    """Ô nghiên cứu: Phonetic Modeling of Dialectal Variation in Vietnamese Speech (arXiv 2605.24451), chờ mã."""

    def __init__(self, **_):
        raise NotImplementedError(
            "arXiv 2605.24451 chưa công bố mã và trọng số. Khi có, cài transcribe() theo giao diện ASR. "
            "Ý tưởng âm vị theo phương ngữ đã có bản dùng được ở asr_post: phonetic_entity_fix.")
