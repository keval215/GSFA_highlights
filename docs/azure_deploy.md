# Azure Deployment Guide — GSFA Highlights VM

**Target VM:** gsfa-highlights (Standard NC4as T4 v3)
**OS:** Ubuntu 24.04 LTS
**GPU:** 1x NVIDIA T4 (16 GB VRAM)
**Public IP:** 4.186.40.179 (dynamic — see Network section)
**Subscription:** Azure subscription 1 (`a56f6649-7402-4bcb-b7df-c5b74d70d180`)
**Resource group:** GSFAStatus / Central India Zone 1

---

## 0. Connect to the VM

```bash
ssh azureuser@4.186.40.179
```

All commands below run on the VM as `azureuser` (sudo as needed).

---

## 1. System Bootstrap

Run once immediately after first login.

```bash
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y \
    git \
    curl \
    wget \
    ca-certificates \
    gnupg \
    lsb-release \
    build-essential \
    unzip \
    htop \
    ncdu \
    jq
```

---

## 2. NVIDIA Driver + CUDA Toolkit

The T4 requires driver >= 525. Install from the official CUDA keyring so the driver and toolkit stay in sync.

### 2a. Remove any pre-installed stubs

```bash
sudo apt-get remove -y --purge '*nvidia*' '*cuda*' 2>/dev/null || true
sudo apt-get autoremove -y
```

### 2b. Add the CUDA 12.4 keyring (matches the Dockerfile base image)

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
```

### 2c. Install driver + CUDA runtime

```bash
# Driver 550 is the current recommended stable for T4 on Ubuntu 24.04
sudo apt-get install -y cuda-drivers-550

# Optional: full CUDA 12.4 toolkit (only needed if compiling CUDA code on the VM)
# sudo apt-get install -y cuda-toolkit-12-4
```

### 2d. Reboot and verify

```bash
sudo reboot
# after reconnect:
nvidia-smi
```

Expected output: T4 listed, driver version ~550.x, CUDA Version 12.4.

---

## 3. Docker + NVIDIA Container Toolkit

The entire pipeline runs inside the Docker image defined in `Dockerfile`. Docker compose handles GPU passthrough.

### 3a. Install Docker Engine (official repo, not snap)

```bash
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

### 3b. Add your user to the docker group (avoids sudo for every docker command)

```bash
sudo usermod -aG docker $USER
newgrp docker          # activate immediately without logout
docker run hello-world # verify
```

### 3c. Install NVIDIA Container Toolkit

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 3d. Verify GPU passthrough

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

---

## 4. Storage Layout (64 GB OS Disk)

The 64 GB disk is sufficient for this workload given that job working directories are cleaned up after each run (see `server.py` `_sweep_stale_work_dirs`). Uploaded match footage is transient — the browser uploads to `/mnt/data/jobs/<job_id>/`, the pipeline runs, and the directory is deleted on completion. Plan for ~15 GB headroom per concurrent job (current `_job_sem` allows only one at a time).

If you find yourself accumulating video on disk, Azure Blob Storage is the clean offload target for raw footage archives (see Section 8).

### Recommended layout

```
/                        ~10 GB   OS + Docker images
/var/lib/docker          ~15 GB   Container layers (including the CUDA base ~6 GB)
/mnt/data/jobs           ~30 GB   Working dir for in-flight video jobs (volume-mounted)
```

### Create the working directory

```bash
sudo mkdir -p /mnt/data/jobs
sudo chown azureuser:azureuser /mnt/data/jobs
```

### Monitor disk usage

```bash
df -h /                  # overall disk
du -sh /mnt/data/jobs/*  # per-job dirs (should be empty when idle)
du -sh /var/lib/docker   # Docker image cache
```

If Docker image cache grows over time, prune old images:

```bash
docker image prune -f
```

---

## 5. Environment File

`docker-compose.yml` sources `/etc/gsfa-highlights.env` for secrets and runtime flags.

```bash
sudo tee /etc/gsfa-highlights.env > /dev/null <<'EOF'
# Azure Blob Storage — required for upload_and_sas in webapp/blob.py
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...;EndpointSuffix=core.windows.net

# GPU flag — set to 1 since the T4 is available
USE_GPU=1

# Safety cap: reject uploads larger than this many gigabytes
MAX_UPLOAD_GB=15
EOF

sudo chmod 600 /etc/gsfa-highlights.env
```

Obtain the `AZURE_STORAGE_CONNECTION_STRING` from the Azure portal: Storage account -> Access keys -> Connection string.

---

## 6. Clone the Repo (PAT-based HTTPS)

### Option A — HTTPS with PAT (your current plan)

```bash
git clone https://<YOUR_GITHUB_USERNAME>:<YOUR_PAT>@github.com/<ORG>/GSFA_highlights.git \
    /opt/gsfa-highlights/repo
```

Do not hardcode the PAT in scripts. Store it in the Git credential helper instead:

```bash
git config --global credential.helper store
# On first clone/pull, Git writes to ~/.git-credentials — PAT is stored in plaintext.
# Acceptable on a single-tenant VM with a locked-down NSG; rotate PAT periodically.
```

### Option B — SSH deploy key (cleaner for CI/CD)

```bash
ssh-keygen -t ed25519 -C "gsfa-highlights-vm" -f ~/.ssh/gsfa_deploy -N ""
cat ~/.ssh/gsfa_deploy.pub
# Add the public key to GitHub repo -> Settings -> Deploy keys (read-only is sufficient for pull)
```

Then clone:

```bash
GIT_SSH_COMMAND='ssh -i ~/.ssh/gsfa_deploy' \
    git clone git@github.com:<ORG>/GSFA_highlights.git /opt/gsfa-highlights/repo
```

---

## 7. Large-Upload Configuration

The webapp now accepts direct file uploads up to `MAX_UPLOAD_GB` (default 15 GB). If you put Nginx in front of Uvicorn (recommended for public exposure), it must be configured to accept large bodies and not buffer them to disk.

Example `nginx` server block:

```nginx
server {
    listen 80;
    server_name _;

    client_max_body_size 15G;         # match MAX_UPLOAD_GB
    client_body_timeout 1800s;        # 30 min — accommodates slow uplinks
    proxy_request_buffering off;      # stream the body straight to FastAPI
    proxy_read_timeout 1800s;
    proxy_send_timeout 1800s;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

`proxy_request_buffering off` is critical — without it nginx buffers the entire 10 GB body to its own disk before forwarding, doubling I/O and stalling the start of the response.

If you expose Uvicorn on port 8000 directly (no nginx), no extra config is needed — FastAPI streams `UploadFile` to a temp file by default.

---

## 8. Build and Start the Stack

```bash
cd /opt/gsfa-highlights/repo

# First run: builds the Docker image (~10-15 min, downloads CUDA base + torch cu124)
docker compose up -d --build

# Verify
docker compose logs -f highlights     # watch startup logs
docker compose ps                     # should show "running"
curl http://localhost:8000/health     # {"ok":true,"use_gpu":true,...}
```

The container exposes port 8000. Uvicorn serves the FastAPI app with a single worker (serialized GPU access is handled in `server.py` via `_job_sem`).

---

## 9. Network / Security

### NSG inbound rules (configure in Azure portal or CLI)

| Priority | Name | Port | Protocol | Source | Action |
|----------|------|------|----------|--------|--------|
| 100 | SSH | 22 | TCP | Your IP or range | Allow |
| 200 | WebApp | 8000 | TCP | Your IP or range | Allow |
| 65000 | DenyAll | * | * | * | Deny |

**Do not open port 8000 to `0.0.0.0/0` unless you add authentication to the FastAPI app first.** The current `server.py` has no auth layer. Restricting to your office/home IP range is sufficient for v1.

If you want to expose the webapp publicly, add HTTP Basic Auth or an API key header check before opening port 8000 to the internet.

To add an NSG rule via Azure CLI:

```bash
az network nsg rule create \
  --resource-group GSFAStatus \
  --nsg-name <your-nsg-name> \
  --name AllowWebApp \
  --priority 200 \
  --destination-port-ranges 8000 \
  --access Allow \
  --protocol Tcp \
  --source-address-prefixes <YOUR_IP>/32
```

### Static / reserved public IP (recommended)

The current public IP (4.186.40.179) is dynamic and will change on VM deallocation. To pin it:

1. Azure portal -> Public IP address resource -> Configuration -> Assignment: Static.
2. Cost is negligible (~$0.004/hr when attached to a running VM; free when the VM is running).

Without a static IP, update your SSH config and any downstream integrations after every deallocate/start cycle.

### Optional: DNS label

In the Azure portal, Public IP -> Configuration -> DNS name label. This gives a stable hostname like `gsfa-highlights.centralindia.cloudapp.azure.com` that survives IP reassignment.

---

## 10. Azure Blob Storage (output destination + video archive)

`webapp/blob.py` uses `AZURE_STORAGE_CONNECTION_STRING` to upload completed highlight reels and return a SAS URL. You need a storage account with a container before the first job succeeds.

```bash
# Create storage account (if not already done)
az storage account create \
  --name gsfastorage \
  --resource-group GSFAStatus \
  --location centralindia \
  --sku Standard_LRS

# Create the output container
az storage container create \
  --name highlights \
  --account-name gsfastorage \
  --public-access off
```

Uploaded match footage lives on the VM only transiently — the browser uploads to `/mnt/data/jobs/<job_id>/`, the pipeline runs, and the directory is deleted on completion. Only the final highlight reel is persisted in Blob.

---

## 11. Deploy Updates (git pull + rebuild)

After pushing new code to GitHub:

```bash
cd /opt/gsfa-highlights/repo
git pull
docker compose up -d --build
```

The `docker-compose.yml` uses `restart: unless-stopped`, so the container comes back automatically after a rebuild.

For automated deploys, the `.github/` directory in the repo already has a GitHub Actions workflow. Ensure the VM's SSH key or PAT is stored as a GitHub Actions secret.

---

## 12. Run as systemd Service (alternative to compose restart policy)

`docker-compose restart: unless-stopped` already handles auto-start on Docker daemon restart. If you want the Docker daemon itself to start on boot (it does by default after install), verify:

```bash
sudo systemctl is-enabled docker   # should print "enabled"
sudo systemctl is-enabled containerd
```

If you prefer a systemd unit for the compose stack directly:

```bash
sudo tee /etc/systemd/system/gsfa-highlights.service > /dev/null <<'EOF'
[Unit]
Description=GSFA Highlights Docker Compose Stack
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/gsfa-highlights/repo
ExecStart=/usr/bin/docker compose up -d --build
ExecStop=/usr/bin/docker compose down
TimeoutStartSec=300

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable gsfa-highlights
sudo systemctl start gsfa-highlights
```

---

## 13. Logs and Monitoring

```bash
# Live container logs
docker compose logs -f highlights

# Last 100 lines
docker compose logs --tail=100 highlights

# Disk usage snapshot
df -h && du -sh /mnt/data/jobs /var/lib/docker

# GPU utilization while a job is running
watch -n 2 nvidia-smi
```

For persistent log retention, redirect Docker logs to a file or enable the Azure Monitor Agent (optional — adds ~$0.03/GB ingested to costs).

---

## 14. Cost Management

| Scenario | Approx. Cost |
|----------|-------------|
| VM running 24/7 | ~$0.53/hr x 720 hr = ~$380/month |
| VM deallocated (stopped) | ~$0.01/hr (OS disk only) |
| VM running during match processing only | $0.53/hr x actual hours |

**Deallocate the VM when not processing matches:**

```bash
# From your local machine (Azure CLI):
az vm deallocate --resource-group GSFAStatus --name gsfa-highlights

# Start it again:
az vm start --resource-group GSFAStatus --name gsfa-highlights
```

Alternatively, set an auto-shutdown schedule in the Azure portal: VM -> Auto-shutdown.

---

## 15. Backup and Snapshots

Snapshot the OS disk before major changes (driver upgrades, OS updates):

```bash
az snapshot create \
  --resource-group GSFAStatus \
  --name gsfa-highlights-snap-$(date +%Y%m%d) \
  --source $(az vm show -g GSFAStatus -n gsfa-highlights \
             --query "storageProfile.osDisk.managedDisk.id" -o tsv)
```

Highlight reels are already uploaded to Blob Storage on job completion (`webapp/blob.py`), so the VM disk is not the source of truth for outputs.

---

## 16. Optional Next Steps

- **Managed identity for Blob access** — replace the connection string with a system-assigned managed identity so there are no secrets in `/etc/gsfa-highlights.env`. Requires granting the VM's identity `Storage Blob Data Contributor` on the storage account.
- **Azure Container Registry (ACR)** — push the Docker image to ACR so deploys pull a pre-built image instead of rebuilding from source on the VM. Speeds up deploys from ~15 min to ~2 min.
- **Nginx reverse proxy** — put Nginx in front of Uvicorn on port 80/443 if you expose the webapp publicly. Handles TLS termination, rate limiting, and request buffering.
- **Azure Monitor Agent + Log Analytics** — structured log ingestion and alerting (e.g., alert if disk > 80%).

---

## Quick-Start Checklist

- [ ] SSH into VM, run system bootstrap (Section 1)
- [ ] Install NVIDIA driver 550 + reboot, verify `nvidia-smi` (Section 2)
- [ ] Install Docker Engine + NVIDIA Container Toolkit, verify GPU passthrough (Section 3)
- [ ] Create `/mnt/data/jobs` (Section 4)
- [ ] Write `/etc/gsfa-highlights.env` with Blob connection string, `USE_GPU=1`, `MAX_UPLOAD_GB=15` (Section 5)
- [ ] Clone repo with PAT or deploy key (Section 6)
- [ ] Configure nginx for large uploads if exposing publicly (Section 7)
- [ ] Create Azure storage account + `highlights` container (Section 10)
- [ ] `docker compose up -d --build`, verify `/health` endpoint (Section 8)
- [ ] Lock down NSG — SSH + port 8000 to your IP only (Section 9)
- [ ] Convert public IP to Static (Section 9)
- [ ] Set VM auto-shutdown or deallocate when idle (Section 14)
