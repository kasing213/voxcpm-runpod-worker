# voxcpm-runpod-worker

RunPod Serverless worker for [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) voice cloning
(Khmer + English). It bills per second only while generating, and scales to zero when idle.

- `handler.py`: the worker. The reference clip is sent with each request, so the image has no voice data.
- `Dockerfile`: torch 2.8 / CUDA 12.8 with the model baked in. GitHub Actions builds it to
  `ghcr.io/kasing213/voxcpm-runpod-worker` on every push to `main`.
- `deploy.py`: creates or updates the RunPod template + endpoint (`python deploy.py <tag>`).

Request:

```json
{"input": {"text": "...", "prompt_wav_b64": "<base64 wav, 5-12 s>", "prompt_text": "<exact words in the clip>",
           "inference_timesteps": 30, "gap": 0.4}}
```

Reply: `{"audio_b64", "sample_rate": 48000, "duration", "sentences"}`.
