# Live AI Simulation (FastAPI + Dhan + Gemini) on GCP (T4)

This guide deploys the `raggiroti` **paper-trading** application:

- Dhan MarketFeed websocket -> ticks
- Builds **1-minute OHLC**
- Runs **Gemini per candle** using your SL-hunting rulebook (RAG retrieval)
- **Simulation only** (no real orders)

## 1) Local run (dev)

From the repo root:

```bash
py -m pip install -r requirements.txt
py -m uvicorn raggiroti.web.app:app --reload --port 8080
```

Open:

`http://localhost:8080/`

Then:
- Save **Dhan client id + access token**
- Save **Gemini API key + models** (decision model + extraction model)
- Start live simulation with `NIFTY` or `BANKNIFTY` (auto-seeds previous trading day levels by default)

### Quick LLM test (no Dhan, no websocket)

You can test per-candle Gemini decisions by calling:

`POST /api/sim/candle`

with a single 1m candle JSON payload.

## 1.1 Previous-day seeding (important)

On **Live Start**, the backend fetches the **previous trading day's 1m candles** using Dhan historical API and computes:

- `prev_close`, `PDH`, `PDL`, `last_hour_high/low`

This improves SL-hunting decisions immediately after open.

## 2) What you need from Dhan

This app uses **DhanHQ-py**:
- MarketFeed websocket (streaming ticks/quotes)
- Your access token generated in Dhan app

Note:
- For **index spot** (NIFTY/BANKNIFTY), OI is not part of the spot feed.
- If you want **option-chain OI**, push snapshots into `/api/live/oi` or add an OI poller.

## 2.1 Rulebook learning pipeline (RAG memory)

The dashboard includes:

- Ingest transcript → store to SQLite
- Extract rules via Gemini → creates a **draft** proposal
- Approve/reject proposal
- Merge approved proposal into `rulebook/nexus_ultra_v2.rulebook.json` (auto assigns new DT-SL ids)

After merging, the backend also rebuilds a **SQLite rulebook index** for faster deterministic retrieval:

- `POST /api/rag/reindex` (manual trigger from UI)

Note: this is **not** a vector/embeddings index; it is a compact cache of the rulebook rules by version.

## 2.2 Live levels (liquidity map)

For quick human-readable levels derived from current live state:

- `GET /api/live/levels`

This includes PDH/PDL, last confirmed swing highs/lows (1m + 5m), and option-chain OI walls (if available).

Endpoints:

- `POST /api/transcripts`
- `POST /api/rule_proposals/extract`
- `POST /api/rule_proposals/{id}/status`
- `POST /api/rule_proposals/{id}/merge`

## 3) Build + run with Docker

```bash
docker build -t raggiroti:latest .
docker run --rm -p 8080:8080 raggiroti:latest
```

Data is stored in SQLite by default at:
`./data/raggiroti.sqlite`

In production, mount a persistent volume to `/app/data`.

## 4) Deploy to Google Compute Engine (T4)

Gemini calls run on Google servers, so **GPU is not required** for Gemini.
You asked for T4; you can still run on a T4 VM (and later add local Llama on the same VM).

### 4.1 Create a VM with T4

Use a region/zone where T4 is available.

Example (adjust zone/machine-type):

```bash
gcloud compute instances create raggiroti-live \
  --zone=asia-south1-b \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --maintenance-policy=TERMINATE \
  --restart-on-failure \
  --boot-disk-size=50GB \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --tags=http-server
```

Open firewall for port 8080 (or put behind a reverse proxy):

```bash
gcloud compute firewall-rules create allow-8080 \
  --allow tcp:8080 \
  --target-tags=http-server
```

### 4.2 Install Docker on the VM

SSH in:

```bash
gcloud compute ssh raggiroti-live --zone=asia-south1-b
```

Install Docker (Debian):

```bash
sudo apt-get update
sudo apt-get install -y docker.io
sudo usermod -aG docker $USER
newgrp docker
```

### 4.3 (Optional) Install NVIDIA drivers

Only needed if you later run a **local GPU model** (Llama/vLLM).

### 4.4 Run the app

Copy the repo to the VM (git clone or scp), then:

```bash
docker build -t raggiroti:latest .
docker run -d --restart=always \
  -p 8080:8080 \
  -v $HOME/raggiroti-data:/app/data \
  --name raggiroti \
  raggiroti:latest
```

Open:
`http://<VM_EXTERNAL_IP>:8080/`

## 5) Performance tips (important)

- Keep LLM prompts compact (the code already sends `state + top rules`).
- Use one uvicorn worker for consistent in-memory state.
- If Gemini latency varies, the engine will still process candles, but decisions will arrive with delay.
- For true low-latency, consider running a **local Llama** on the T4 and using Gemini only for offline rule extraction.
