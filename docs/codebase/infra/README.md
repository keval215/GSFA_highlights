# Infrastructure, deployment & tests

How the service is packaged, deployed, configured, and tested.

| File | Role |
|---|---|
| `Dockerfile` | Single image for both `api` and `worker` |
| `docker-compose.yml` | Runs `api` (CPU) + `worker` (GPU) from that image |
| `.github/workflows/deploy.yml` | CI/CD: push to `main` → SSH redeploy on the Azure VM |
| `requirements.txt` | Local dev dependencies |
| `requirements-service.txt` | Service (Docker image) dependencies |
| `tests/` | Automated unit tests (`test_stats.py`, `test_db.py`, `test_notifier.py`) |
| `docs/azure_deploy.md` | Human deploy/runbook for the VM |
| `docs/API.md` | HTTP API + environment variable reference |

---

## `Dockerfile`
- Base: `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` (T4 VM, CUDA 12.4).
- **Installs stable Python 3.11 from deadsnakes**, *not* Ubuntu 22.04's own
  `python3.11` — which is `3.11.0rc1`, a release candidate that has been observed to
  segfault `torch.jit.script`. This is a deliberate, load-bearing choice; do not switch to
  the system interpreter. (Originally driven by an rfdetr crash; rfdetr is gone but the
  reasoning — never ship a pre-release interpreter — stands.)
- Builds an isolated venv at `/opt/venv` so deadsnakes 3.11 never picks up Ubuntu's apt
  python3.10 packages (which would break `azure-storage-blob` via a mismatched
  `_cffi_backend`).
- Installs ODBC Driver 18 (`msodbcsql18`) for pyodbc and the OpenCV native libs
  (`libgl1`, `libglib2.0-0`, …) that ultralytics pulls in even headless.
- Torch/torchvision from the cu124 index first (big, rarely-changing layer), then
  `requirements-service.txt`, then copies the source packages.
- Default `CMD` is the worker; the api command is overridden in compose.

## `docker-compose.yml`
- `api` — `uvicorn service.api:app` on port 8000, no GPU, mounts `/mnt/data`.
- `worker` — `python -m service.worker`, `gpus: all`, mounts `/mnt/data` and the
  read-only models dir `/opt/gsfa-highlights/models`. Sets `HF_HOME=/mnt/data/hf_cache`
  so SigLIP weights/`spiece.model` survive container rebuilds (the team-fit pkl references
  these files by path; an ephemeral cache would break the pkl after every rebuild).
- Both load secrets/config from **`env_file: /etc/gsfa-highlights.env`** on the VM.

> **Config gotcha:** every variable in `/etc/gsfa-highlights.env` must be `NAME=value`,
> with the value being the bare value. Writing `PLAYER_WEIGHTS=PLAYER_WEIGHTS=/path/...`
> (name duplicated into the value) makes the worker try to load a model file literally
> named `PLAYER_WEIGHTS=/path/...` and crash at startup with `FileNotFoundError`.

## `.github/workflows/deploy.yml`
- Trigger: push to **`main`** (or manual `workflow_dispatch`). Previously watched `test`;
  retargeted to `main` so the repo's main branch is what deploys.
- Action: SSH to the VM, `git reset --hard origin/main`, remove any legacy standalone
  container, `docker compose up -d --build --remove-orphans`, prune old images.
- Requires repo secrets: `AZURE_VM_HOST`, `AZURE_VM_USER`, `AZURE_VM_SSH_KEY`,
  optional `AZURE_VM_PORT`.
- Expected VM layout: repo at `/opt/gsfa-highlights/repo`, env at
  `/etc/gsfa-highlights.env`, models at `/opt/gsfa-highlights/models`, Docker +
  nvidia-container-toolkit installed.

## Requirements
- `requirements-service.txt` — the image's deps: CV stack (opencv-headless, ultralytics,
  supervision, boxmot, umap-learn, scikit-learn, transformers, sentencepiece, the
  `roboflow/sports` package) + service (fastapi, uvicorn, azure-storage-blob/queue,
  pyodbc, requests). torch/torchvision are installed separately in the Dockerfile.
- `requirements.txt` — local dev (includes `opencv-python` full, `paddleocr`, `tqdm`,
  etc.) for the standalone scripts.

## `tests/`
Automated unit tests targeting the dependency-free correctness logic:
- `test_stats.py` — per-minute counting, adjustments, `split_adjustment`, `is_expected`,
  `build_payload`. This is where the possession-math invariants are pinned.
- `test_db.py` — DB-layer behaviour.
- `test_notifier.py` — outbox/callback behaviour.

These avoid torch/boxmot/GPU on purpose (that's why `stats.py` is dependency-free), so
they run fast in CI/local without a GPU.

## Deploy/runbook
`docs/azure_deploy.md` is the human runbook for the VM (bootstrapping
`/etc/gsfa-highlights.env`, etc.). `docs/API.md` is the HTTP + env-var reference.
