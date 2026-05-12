# GSFA Highlights — Azure deployment runbook

Single-VM deployment of the highlight pipeline. The user opens
`https://highlights.yourdomain.com/`, pastes a YouTube URL, and gets a
download link for the generated highlight reel.

## Architecture (one box)

```
Cloudflare DNS (CNAME highlights.yourdomain.com -> vm-...cloudapp.azure.com)
       |
       v
  Azure VM (Standard_NC4as_T4_v3, Ubuntu 22.04)
       |
  nginx (TLS via Let's Encrypt) -> :8000 (FastAPI in Docker)
                                       |
                                       v
                              Azure Blob Storage (highlights container)
                                       |
                                       v
                              Signed SAS URL returned to browser
```

No ACR. No queue. No separate worker. Image is built **on the VM** from a
`git pull` of this repo, so a GitHub Action SSHing in to run
`docker compose up -d --build` is the full deploy cycle.

---

## 1. One-time Azure provisioning

Do this in the Azure Portal (or `az` CLI) once.

### 1a. Resource group
- Name: `rg-gsfa-highlights`
- Region: `centralindia` (or closest GPU-available region)

### 1b. Storage Account
- Name: `stgsfahighlights` (must be globally unique — adjust)
- SKU: Standard_LRS
- Inside, create a **container** named `highlights` (private; we hand out SAS URLs)
- Under "Access keys", copy the **Connection string** — you'll set it as
  `AZURE_STORAGE_CONNECTION_STRING` on the VM.

### 1c. Virtual Machine
- Name: `vm-gsfa-highlights`
- Size: `Standard_NC4as_T4_v3` (1× T4 GPU, 4 vCPU, 28 GB RAM)
- Image: Ubuntu Server 22.04 LTS — x64
- Authentication: SSH public key. Save the **private key** locally; you'll
  add it to GitHub repo secrets as `AZURE_VM_SSH_KEY`.
- OS disk: 128 GB Premium SSD
- **Add a data disk: 512 GB Premium SSD**, format ext4, mount at `/mnt/data`.
  This is the scratch space for video downloads + intermediate clips. A 90-min
  match @1080p is 5–10 GB; do **not** put this on the OS disk.
- Networking / NSG:
  - Allow inbound TCP **22** (SSH) from your IP only
  - Allow inbound TCP **443** (HTTPS) from `Any`
  - Allow inbound TCP **80** (HTTP) from `Any` (Let's Encrypt http-01 challenge)
- **Auto-shutdown**: Enable at e.g. 02:00 UTC daily. Cost control — you can
  always start it manually from the portal when needed.

### 1d. Custom domain
- In Cloudflare, add a **CNAME**:
  `highlights.yourdomain.com  -> vm-gsfa-highlights.<region>.cloudapp.azure.com.`
- Set **Proxy status: DNS only (grey cloud)** initially. Cloudflare's
  orange-cloud proxy interferes with Let's Encrypt http-01; you can switch
  to orange after issuing the cert if you want CF caching/DDoS.

---

## 2. VM bootstrap (one-time, ~15 min)

SSH in: `ssh azureuser@vm-...cloudapp.azure.com`

### 2a. Mount the data disk
```bash
# Find the device (typically /dev/sdc on Azure)
lsblk
sudo mkfs.ext4 /dev/sdc
sudo mkdir -p /mnt/data
sudo mount /dev/sdc /mnt/data
echo "/dev/sdc /mnt/data ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab
sudo mkdir -p /mnt/data/jobs && sudo chown $USER /mnt/data/jobs
```

### 2b. NVIDIA driver + Docker + nvidia-container-toolkit
```bash
# Ubuntu's recommended NVIDIA driver for T4
sudo apt-get update
sudo ubuntu-drivers install --gpgpu
sudo reboot
# wait ~60s, SSH back in. Verify:
nvidia-smi   # should show the T4

# Docker
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
  sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io \
  docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER
newgrp docker   # or log out and back in

# nvidia-container-toolkit so containers see the GPU
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi   # smoke test
```

### 2c. Clone the repo
```bash
sudo mkdir -p /opt/gsfa-highlights
sudo chown $USER /opt/gsfa-highlights
git clone https://github.com/<your-org>/GSFA_highlights.git /opt/gsfa-highlights
cd /opt/gsfa-highlights
```

### 2d. Secrets and env file
```bash
sudo mkdir -p /opt/gsfa-highlights/secrets
sudo chmod 700 /opt/gsfa-highlights/secrets

# YouTube cookies file — see "Preparing the cookies file" below.
sudo cp youtube_cookies.txt /opt/gsfa-highlights/secrets/youtube_cookies.txt
sudo chmod 600 /opt/gsfa-highlights/secrets/youtube_cookies.txt

# Env file
sudo tee /etc/gsfa-highlights.env >/dev/null <<'EOF'
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=stgsfahighlights;AccountKey=...;EndpointSuffix=core.windows.net
BLOB_CONTAINER=highlights
BLOB_SAS_DAYS=7
USE_GPU=1
MAX_VIDEO_DURATION_MIN=120
EOF
sudo chmod 600 /etc/gsfa-highlights.env
```

### 2e. First container build + run
```bash
cd /opt/gsfa-highlights
docker compose up -d --build
docker compose logs -f highlights   # watch boot; press Ctrl-C when ready
curl http://localhost:8000/health   # {"ok": true, ...}
```

### 2f. nginx + Let's Encrypt
```bash
sudo apt-get install -y nginx certbot python3-certbot-nginx

sudo tee /etc/nginx/sites-available/highlights >/dev/null <<'EOF'
server {
    listen 80;
    server_name highlights.yourdomain.com;
    client_max_body_size 200m;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 600s;     # long polls during big jobs
        proxy_send_timeout 600s;
    }
}
EOF
sudo ln -sf /etc/nginx/sites-available/highlights /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx

# Issue TLS cert. Cloudflare must be grey-cloud at this point.
sudo certbot --nginx -d highlights.yourdomain.com --agree-tos -m you@yourdomain.com --redirect
```

Open `https://highlights.yourdomain.com/` — you should see the form. Done.

---

## 3. GitHub auto-deploy

Add these **GitHub repo secrets** (Settings → Secrets and variables → Actions):

| Secret | Value |
|---|---|
| `AZURE_VM_HOST` | `vm-gsfa-highlights.<region>.cloudapp.azure.com` (or IP) |
| `AZURE_VM_USER` | `azureuser` |
| `AZURE_VM_SSH_KEY` | Contents of the PEM private key matching the VM |
| `AZURE_VM_PORT` | Optional, defaults to `22` |

The workflow at `.github/workflows/deploy.yml` runs on every push to `main`:
SSHes in, `git pull`, `docker compose up -d --build`, prunes old images.

---

## 4. Preparing the YouTube cookies file

YouTube increasingly blocks downloads from cloud IP ranges with the
"Sign in to confirm you're not a bot" wall. Solution: feed yt-dlp a cookies
file from a logged-in browser session.

1. In **Chrome or Firefox** install the extension **"Get cookies.txt LOCALLY"**
   (the *LOCALLY* variant is open-source and does not upload your cookies).
2. In that browser, **log into YouTube** (a fresh / dedicated account is
   safer than your main one).
3. While on a YouTube page, click the extension icon → **Export** for the
   current site → it downloads `youtube.com_cookies.txt`.
4. Inspect the file — it should start with `# Netscape HTTP Cookie File`
   and contain rows for `.youtube.com` (look for `SID`, `HSID`, `SSID`,
   `__Secure-3PSID` etc.).
5. Copy it to the VM at `/opt/gsfa-highlights/secrets/youtube_cookies.txt`
   (chmod 600).
6. The container mounts that path read-only at `/secrets/youtube_cookies.txt`
   and `ytdlp_fetch.py` picks it up via `YOUTUBE_COOKIES_FILE`.

**Treat the file like a password.** Anyone with it can post as that
account. Refresh every few weeks; YouTube rotates session cookies.

---

## 5. Operating cheat-sheet

| Action | Command |
|---|---|
| Tail logs | `docker compose logs -f highlights` |
| Restart  | `docker compose restart highlights` |
| Rebuild after manual edit | `docker compose up -d --build` |
| Free disk | `docker image prune -f && rm -rf /mnt/data/jobs/*` |
| Manual GPU check inside container | `docker compose exec highlights nvidia-smi` |
| Stop VM (save money) | Azure Portal → VM → Stop (deallocate) |

---

## 6. Known limitations of this v1 setup

- **One job at a time** (`asyncio.Semaphore(1)`). Multiple submissions queue.
- **Job state is in-memory.** If the container restarts mid-job, that job is lost.
- **No retry.** If yt-dlp fails (cookies expired, video deleted), the job
  ends `failed` with the underlying error message — user must resubmit.
- **No auth.** Anyone who knows the URL can submit a job. Put it behind
  Cloudflare Access or basic auth in nginx if that matters.
