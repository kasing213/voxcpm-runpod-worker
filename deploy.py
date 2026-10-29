"""Create or update the RunPod Serverless endpoint for this worker.

    python deploy.py [image-tag]        (default tag: latest)

Reads RUNPOD_API_KEY from the environment or D:\\Video-testing\\.env, and writes
RUNPOD_VOXCPM_ENDPOINT_ID back into that .env so the video engine can call it.
Safe to re-run: it reuses the template and endpoint it finds by name.
"""
import json
import os
import re
import sys
import urllib.request

API = "https://rest.runpod.io/v1"
NAME = "voxcpm-worker"
IMAGE = "ghcr.io/kasing213/voxcpm-runpod-worker"
ENV_FILE = r"D:\Video-testing\.env"
# 16-24 GB cards; VoxCPM2 runs on an A4000 (16 GB). Several types so one sold-out card never blocks.
GPUS = ["NVIDIA RTX A4000", "NVIDIA RTX A4500", "NVIDIA RTX 4000 Ada Generation",
        "NVIDIA RTX A5000", "NVIDIA GeForce RTX 4090"]


def read_env(key):
    if os.environ.get(key):
        return os.environ[key]
    if os.path.exists(ENV_FILE):
        m = re.search(rf"^{key}=(.*)$", open(ENV_FILE, encoding="utf-8").read(), re.M)
        if m:
            return m.group(1).strip().strip('"')
    sys.exit(f"{key} not set (environment or {ENV_FILE})")


def write_env(key, value):
    text = open(ENV_FILE, encoding="utf-8").read() if os.path.exists(ENV_FILE) else ""
    line = f"{key}={value}"
    text = re.sub(rf"^{key}=.*$", line, text, flags=re.M) if re.search(rf"^{key}=", text, re.M) \
        else text.rstrip("\n") + ("\n" if text else "") + line + "\n"
    open(ENV_FILE, "w", encoding="utf-8").write(text)


def call(method, path, body=None):
    req = urllib.request.Request(
        API + path, method=method,
        data=json.dumps(body).encode() if body is not None else None,
        # Cloudflare in front of RunPod 403s (error 1010) urllib's default User-Agent.
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json",
                 "User-Agent": "voxcpm-worker-deploy/1.0"})
    try:
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read() or "null")
    except urllib.error.HTTPError as e:
        sys.exit(f"{method} {path} -> {e.code}: {e.read().decode()[:500]}")


def find(path):
    r = call("GET", path)
    items = r if isinstance(r, list) else next((v for v in (r or {}).values() if isinstance(v, list)), [])
    return next((i for i in items if i.get("name") == NAME), None)


KEY = read_env("RUNPOD_API_KEY")
image = f"{IMAGE}:{sys.argv[1] if len(sys.argv) > 1 else 'latest'}"

tpl = find("/templates")
if tpl:
    call("POST", f"/templates/{tpl['id']}/update", {"imageName": image})
    print("template updated:", tpl["id"], image)
else:
    tpl = call("POST", "/templates", {
        "name": NAME, "imageName": image, "isServerless": True,
        "containerDiskInGb": 30, "env": {"VOXCPM_OPTIMIZE": "0"}})
    print("template created:", tpl["id"], image)

ep = find("/endpoints")
if not ep:
    ep = call("POST", "/endpoints", {
        "name": NAME, "templateId": tpl["id"], "gpuTypeIds": GPUS,
        "workersMin": 0,          # nothing runs, nothing bills, while idle
        "workersMax": 1,
        "idleTimeout": 60,        # keep a warm worker 60s for the next sentence batch
        "flashboot": True,
        "executionTimeoutMs": 900000})
    print("endpoint created:", ep["id"])
else:
    print("endpoint exists:", ep["id"])

write_env("RUNPOD_VOXCPM_ENDPOINT_ID", ep["id"])
print(f"RUNPOD_VOXCPM_ENDPOINT_ID={ep['id']} written to {ENV_FILE}")
