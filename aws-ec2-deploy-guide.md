# AWS EC2 Deployment — Face Recognition API (FileZilla + Apache reverse proxy + SSL)

FastAPI app, run as a single uvicorn worker under systemd, fronted by Apache (reverse proxy)
for TLS termination. Files are transferred with FileZilla (SFTP, using your `.ppk` key),
package installs and service management are done over SSH.

Reference values — replace with your actual values:
- Domain: hrmsai3857test.site
- Instance: Ubuntu 24.04, Elastic IP already assigned
- SFTP/SSH user: ubuntu (your `.ppk` key)
- Deploy path: /opt/faceapi

---

## STEP 1 — Security group

Inbound rules on the instance:
- 22 (SSH/SFTP) — restrict to your IP
- 80 (HTTP) — 0.0.0.0/0
- 443 (HTTPS) — 0.0.0.0/0

Port 8000 (uvicorn) is **not** opened publicly — Apache is the only thing that talks to it,
over localhost.

---

## STEP 2 — Connect over SSH once (for setup only)

FileZilla handles file transfer; package installs and service commands need a terminal.
Either use FileZilla's own "Open in terminal" if available, or SSH in directly with the
same `.ppk` (via PuTTY, or convert it to OpenSSH format with `puttygen` if you'd rather use
a regular `ssh -i` from a Mac/Linux machine).

```bash
sudo apt update && sudo apt upgrade -y
timedatectl set-timezone 'Asia/Kolkata'
```

---

## STEP 3 — System packages

```bash
sudo apt install -y python3.12-venv python3-pip apache2
```

Ubuntu 24.04 ships Python 3.12, which matches what this project was built/tested against.

---

## STEP 4 — Create the deploy directory

```bash
sudo mkdir -p /opt/faceapi
sudo chown -R ubuntu:ubuntu /opt/faceapi
```

Owned by `ubuntu` so FileZilla (connecting with your `.ppk` as `ubuntu`) can drop files
straight in without permission errors.

---

## STEP 5 — Transfer the project with FileZilla

Connect FileZilla using:
- Protocol: SFTP
- Host: `<Elastic IP>`
- Key file: your `.ppk`
- Username: `ubuntu`

Upload into `/opt/faceapi`:
- `app/` (the whole package)
- `config.toml`
- `requirements.txt`
- `models/` (both `.onnx` files — liveness detection needs them and the app doesn't
  download them on its own)

Do **not** upload your local `.env` — it gets created directly on the server in Step 7, so
the API key never has to sit in a file on your laptop that you're dragging into FileZilla.

Skip: `.venv/`, `data/`, `__pycache__/`, `.git/` — these are either generated on the server
or don't belong on it. `data/faces.db` in particular should not be overwritten by re-uploads
once the app is running — it's the live registered-faces database.

---

## STEP 6 — Python virtualenv and dependencies (SSH)

```bash
cd /opt/faceapi
python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

First run of the app will also have `insightface` download the `buffalo_sc` detection/
recognition bundle to `~/.insightface/` — needs outbound internet the first time only.

---

## STEP 7 — Data directory and environment file (SSH)

```bash
mkdir -p /opt/faceapi/data
chmod 750 /opt/faceapi/data
```

Create `/opt/faceapi/.env` directly on the server:

```bash
nano /opt/faceapi/.env
```
```
API_KEY=<your key>
```
```bash
chmod 600 /opt/faceapi/.env
```

The app reads `API_KEY` from the process environment (`app/api/deps.py`), not from a `.env`
parser — the systemd unit below loads this file with `EnvironmentFile=`, which is what
actually gets it into the process.

---

## STEP 8 — systemd service

```bash
sudo nano /etc/systemd/system/faceapi.service
```

Paste:

```ini
[Unit]
Description=Face Recognition Attendance API
After=network.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/opt/faceapi
EnvironmentFile=/opt/faceapi/.env
ExecStart=/opt/faceapi/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
Restart=on-failure
RestartSec=3

# one worker only — sessions (registration/attendance) live in process memory,
# see app/main.py

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now faceapi
sudo systemctl status faceapi
curl http://127.0.0.1:8000/
```

The `/` route returns `{"service": "face-recognition", "status": "ok", ...}` when the models
loaded correctly — check `sudo journalctl -u faceapi -f` if it doesn't come up.

---

## STEP 9 — Enable Apache proxy modules

```bash
sudo a2enmod proxy
sudo a2enmod proxy_http
sudo a2enmod ssl
sudo systemctl restart apache2
```

---

## STEP 10 — Apache virtual host (reverse proxy)

```bash
sudo nano /etc/apache2/sites-available/faceapi.conf
```

Paste:

```apache
<VirtualHost *:80>
    ServerName hrmsai3857test.site
    ServerAlias www.hrmsai3857test.site

    ProxyPreserveHost On
    ProxyPass / http://127.0.0.1:8000/
    ProxyPassReverse / http://127.0.0.1:8000/

    # uploaded images: match the app's own max_image_mb (default 5) with headroom
    LimitRequestBody 10485760

    ErrorLog ${APACHE_LOG_DIR}/faceapi-error.log
    CustomLog ${APACHE_LOG_DIR}/faceapi-access.log combined
</VirtualHost>
```

```bash
sudo a2ensite faceapi.conf
sudo a2dissite 000-default.conf
sudo apache2ctl configtest
sudo systemctl restart apache2

# confirm it's enabled
ls -la /etc/apache2/sites-enabled/
```

You must see `faceapi.conf` linked in `sites-enabled/` — if this is missed, Certbot may
configure the wrong/default site later.

```bash
curl http://localhost
```

---

## STEP 11 — DNS setup (at your registrar for hrmsai3857test.site)

| Type | Name | Value |
|------|------|-------|
| A | @ | `<EC2 Elastic IP>` |
| A | www | `<EC2 Elastic IP>` |

```bash
nslookup hrmsai3857test.site
curl http://hrmsai3857test.site
```

Wait for DNS to propagate before running Certbot (can take a few minutes to a few hours).

---

## STEP 12 — Install Certbot and get SSL certificate

```bash
sudo apt install certbot python3-certbot-apache -y
sudo certbot --apache -d hrmsai3857test.site -d www.hrmsai3857test.site
```

Certbot edits `/etc/apache2/sites-available/faceapi-le-ssl.conf`, adds the SSL block, and
sets up the HTTP → HTTPS redirect.

---

## STEP 13 — Verify SSL and auto-renewal

```bash
sudo certbot certificates
sudo systemctl list-timers | grep certbot
sudo certbot renew --dry-run
```

---

## STEP 14 — Full end-to-end verification

```bash
sudo systemctl status faceapi
sudo systemctl status apache2
ls -la /etc/apache2/sites-enabled/

curl https://hrmsai3857test.site/
curl -I http://hrmsai3857test.site       # should redirect to https

# an authenticated route — expect 401 without the header, normal response with it
curl -I https://hrmsai3857test.site/api/people
curl -H "X-API-Key: <your key>" https://hrmsai3857test.site/api/people
```

Interactive API docs: `https://hrmsai3857test.site/docs`.

---

## Connecting a frontend to this API

`app/main.py` currently sets `allow_origins=["*"]` for CORS — fine for testing, but once the
calling frontend's origin is known, narrow it to that origin(s) in `app/main.py` and
redeploy, rather than leaving it wide open in production.

---

## ONGOING — Deploying an update

1. FileZilla: re-upload only the changed files into `/opt/faceapi` (still skip `data/`,
   `.venv/`, `.env`).
2. SSH in:
```bash
cd /opt/faceapi
.venv/bin/pip install -r requirements.txt   # only if requirements.txt changed
sudo systemctl restart faceapi
sudo systemctl status faceapi
```

---

## TROUBLESHOOTING

| Symptom | Likely cause | Check |
|---|---|---|
| 502 Bad Gateway from Apache | uvicorn/faceapi service not running | `sudo systemctl status faceapi`, `sudo journalctl -u faceapi -e` |
| App fails to start, model errors | `models/*.onnx` missing or wrong path | Confirm `models/MiniFASNetV1SE.onnx` and `MiniFASNetV2.onnx` exist under `/opt/faceapi/models` |
| Every `/api/*` call returns 401 | `API_KEY` not reaching the process | Confirm `/opt/faceapi/.env` exists, `EnvironmentFile=` is in the unit, `sudo systemctl daemon-reload` after editing |
| First request very slow / times out | insightface downloading model bundle on first run, no internet egress | Check security group / NAT allows outbound HTTPS, or pre-seed `~/.insightface/` |
| SSL cert fails to issue | DNS not propagated yet | `nslookup hrmsai3857test.site` before running Certbot |
| Site shows Apache default page | Virtual host not enabled, or default site still enabled | `ls -la /etc/apache2/sites-enabled/` |
| `data/faces.db` got overwritten by a re-upload | FileZilla re-upload included `data/` | Exclude `data/` from future uploads; restore from your last backup |
