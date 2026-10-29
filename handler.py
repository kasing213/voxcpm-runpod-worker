"""RunPod Serverless worker: VoxCPM2 voice cloning.

Input  {"text", "prompt_wav_b64", "prompt_text", "inference_timesteps"?, "gap"?}
Output {"audio_b64" (wav), "sample_rate", "duration", "sentences"}

The reference clip travels in each request, so this image holds no voice data and
can be public. Splitting and joining is the proven recipe from ref/batch_generate.py.
"""
import base64
import io
import os
import re
import tempfile

import numpy as np
import runpod
import soundfile as sf
from voxcpm import VoxCPM

EDGE_FADE = 0.015
MAX_CHARS = 4000            # ~3 min of speech; keeps the base64 reply under RunPod's limit
KHMER = lambda s: any("ក" <= c <= "៿" for c in s)

# generate() never denoises by default, so skip loading the denoiser (no second download).
# torch.compile makes each step faster but costs minutes on every cold start: off by default.
MODEL = VoxCPM.from_pretrained(
    "openbmb/VoxCPM2",
    load_denoiser=False,
    optimize=os.environ.get("VOXCPM_OPTIMIZE") == "1",
)
SR = int(MODEL.tts_model.sample_rate)  # 48000, never hardcode


def gap_bed(n, rms):
    """Gap filler at the take's own noise floor; digital zero makes the room tone click off."""
    z = np.random.standard_normal(n).astype(np.float32)
    z = np.convolve(z, np.ones(64, dtype=np.float32) / 64, mode="same")
    return z / (float(np.sqrt((z ** 2).mean())) + 1e-12) * rms


def edge_fade(x):
    k = min(int(EDGE_FADE * SR), len(x) // 2)
    if k < 2:
        return x
    w = (0.5 * (1 - np.cos(np.pi * np.arange(k) / (k - 1)))).astype(np.float32)
    x = np.array(x, dtype=np.float32, copy=True)
    x[:k] *= w
    x[-k:] *= w[::-1]
    return x


def floor_rms(x):
    n = int(0.02 * SR)
    if len(x) < n * 5:
        return float(np.sqrt((x ** 2).mean()) * 0.05)
    fr = x[: len(x) // n * n].reshape(-1, n)
    return float(np.percentile(np.sqrt((fr ** 2).mean(1)), 5))


def split_sentences(text):
    if KHMER(text):
        parts = [s.strip() + "។" for s in text.split("។") if s.strip()]
    else:
        parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return parts or [text]


def handler(job):
    inp = job["input"]
    text = (inp.get("text") or "").strip()
    prompt_text = (inp.get("prompt_text") or "").strip()
    if not text or not prompt_text or not inp.get("prompt_wav_b64"):
        return {"error": "text, prompt_wav_b64 and prompt_text are required"}
    if len(text) > MAX_CHARS:
        return {"error": f"text is {len(text)} chars; split it (max {MAX_CHARS})"}
    steps = int(inp.get("inference_timesteps", 30))
    gap = float(inp.get("gap", 0.40))

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(base64.b64decode(inp["prompt_wav_b64"]))
        ref_path = f.name
    try:
        sentences = split_sentences(text)
        pieces = [
            edge_fade(np.asarray(
                MODEL.generate(text=s, prompt_wav_path=ref_path, prompt_text=prompt_text,
                               inference_timesteps=steps),
                dtype=np.float32))
            for s in sentences
        ]
    finally:
        os.unlink(ref_path)

    bed = float(np.median([floor_rms(p) for p in pieces]))
    joined = [pieces[0]]
    for p in pieces[1:]:
        joined += [gap_bed(int(gap * SR), bed), p]
    audio = np.concatenate(joined)
    audio += gap_bed(len(audio), bed * 0.5)  # continuous floor, nothing snaps

    buf = io.BytesIO()
    sf.write(buf, audio, SR, format="WAV", subtype="PCM_16")
    return {
        "audio_b64": base64.b64encode(buf.getvalue()).decode(),
        "sample_rate": SR,
        "duration": round(len(audio) / SR, 2),
        "sentences": len(sentences),
    }


runpod.serverless.start({"handler": handler})
