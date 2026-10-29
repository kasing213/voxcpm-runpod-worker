# torch 2.8 + CUDA 12.8 is the stack proven on the RunPod pods (VOXCPM2_RUNBOOK.md).
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# runpod 1.10: 1.11+ needs tomlkit>=0.15.1, which voxcpm's gradio 6 forbids (<0.14).
RUN pip install --no-cache-dir torch==2.8.0 torchaudio==2.8.0 torchcodec==0.7.0 \
    && pip install --no-cache-dir voxcpm==2.0.3 runpod==1.10.0

# Bake the ~5 GB model into the image so a cold start loads from disk, not Hugging Face.
ENV HF_HOME=/models HF_HUB_DISABLE_XET=1
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('openbmb/VoxCPM2')"

COPY handler.py /handler.py
CMD ["python", "-u", "/handler.py"]
