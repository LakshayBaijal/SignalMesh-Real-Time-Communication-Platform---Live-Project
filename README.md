# SignalMesh — Real-Time Communication Platform (Live Project)

**Live demo:** https://signal-mesh.onrender.com/

SignalMesh is a real-time chat platform built with **FastAPI + WebSockets** with authentication, presence, and scalable full‑duplex client/server messaging.

---

## Key Features
- Real-time global chat via **WebSockets** (full-duplex)
- **JWT authentication** + **bcrypt** password hashing
- Presence tracking (online/offline)
- File attachments + emoji support
- Broadcast messaging architecture
- Deploy-ready configuration (environment based) for Render

---

## Tech Stack
**Backend:** FastAPI, WebSockets, SQLAlchemy, SQLite, JWT, bcrypt  
**Frontend:** JavaScript, HTML, CSS  
**Deployment:** Render

---

## Quick Start (Local)

### 1) Clone
```bash
git clone https://github.com/LakshayBaijal/SignalMesh-Real-Time-Communication-Platform---Live-Project.git
cd SignalMesh-Real-Time-Communication-Platform---Live-Project
```

### 2) Backend setup (Python)
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3) Run the server
```bash
uvicorn server:app --reload
```

Open: `http://127.0.0.1:8000`

---

## Configuration (Environment Variables)
Create a `.env` (or configure on Render) with values like:

- `JWT_SECRET` — secret for signing tokens
- `DATABASE_URL` — DB connection string (optional; defaults may be used)
- Any Render-specific variables required by your deployment

> Tip: Never commit real secrets. Use Render env vars for production.

---

## API / Healthcheck
Common endpoints (adjust if your backend differs):
- `GET /ping` — basic healthcheck

---

# Benchmarking & Performance Metrics (k6)

This repo includes **reproducible load-testing scripts** under `benchmarks/` to measure HTTP latency, error rate, WebSocket connection time, and message round-trip latency.

## What’s Included
`benchmarks/` contains:
- `http_load.js` — HTTP latency/error rate for `/ping` + login throughput
- `ws_load.js` — authenticated WebSocket connect time + message RTT
- `http_stress.js`, `ws_stress.js` — higher-load stress scenarios (optional)
- Output examples: `*_load.json`, `*_load.log` (sample runs)

## Install k6
**Windows (PowerShell):**
```powershell
winget install k6.k6
```
or
```powershell
choco install k6
```

## Run Benchmarks (Recommended: 3 runs each)
From the project root:

### 1) Set benchmark environment variables (example)
```powershell
$env:BASE_URL = "https://signal-mesh.onrender.com"
$env:BENCH_USERNAME = "bench_login_user"
$env:BENCH_PASSWORD = "BenchPass123!"
$env:WS_HOLD_SECONDS = "15"
$env:MESSAGE_INTERVAL_SECONDS = "3"
```

### 2) HTTP benchmark (latency + login throughput)
```powershell
k6 run .\benchmarks\http_load.js --out json=.\benchmarks\http_load.json *> .\benchmarks\http_load.log
```

### 3) WebSocket benchmark (connect + RTT)
```powershell
k6 run .\benchmarks\ws_load.js --out json=.\benchmarks\ws_load.json *> .\benchmarks\ws_load.log
```

### 4) (Optional) Stress tests
```powershell
k6 run .\benchmarks\http_stress.js --out json=.\benchmarks\http_stress.json *> .\benchmarks\http_stress.log
k6 run .\benchmarks\ws_stress.js --out json=.\benchmarks\ws_stress.json *> .\benchmarks\ws_stress.log
```

---

## Reported Metrics (Real Sample From Repo Runs)

Below are **sample metrics already present in this repository’s benchmark outputs** (generated on **2026-05-14**). Treat them as *environment-dependent* (Render free tier varies by region/time), and re-run to publish your “official” numbers.

### HTTP (k6 `http_load.js`)
From `benchmarks/http_load.log`:
- **Total requests:** `50`
- **Request rate:** `~1.166 req/s`
- **HTTP request duration:** `avg ~650 ms`, `med ~310 ms`, `min ~282 ms`, `max ~2.19 s`
- **HTTP request failed:** `~2%` (1 failed out of 49–50 visible in summary)

**Interpretation:** Under light load on a free-tier deployment, `/ping` tends to sit around ~0.3–0.4s median latency, with occasional slower spikes.

### WebSocket (k6 `ws_load.js`)
From `benchmarks/ws_load.log`:
- **WebSocket connect time (`ws_connect_time_ms`):** `avg ~1.06 s` (min ~998 ms, max ~1.14 s)
- **WebSocket round-trip (`ws_roundtrip_ms`):** `avg ~338 ms` (min ~299 ms, max ~388 ms)
- **Messages sent:** `8`
- **Messages received:** `23`
- **WS checks:** `100%`

**Interpretation:** Authenticated WS connect takes about ~1s on the deployed environment; round-trip latency for chat messages is ~0.3s average in this sample.

---

## How to Present Metrics on Your Resume / Project Page (Template)
Use only metrics visible in k6 output and state the environment:

- “Benchmarked deployed FastAPI + WebSocket service using **k6** (Render free tier) and observed:
  - HTTP latency (p50/p95/p99) and error rate for `/ping` and login flows
  - WebSocket connect time and message round-trip latency under concurrent sessions”

> Best practice: run each scenario **3 times** and report the most stable run (or median of runs), plus your environment details (region, tier, time window).

---

## Project Structure (Suggested)
A typical layout (update to match your repo):
```text
.
├── server.py / app.py / main.py
├── requirements.txt
├── benchmarks/
│   ├── http_load.js
│   ├── ws_load.js
│   ├── http_stress.js
│   ├── ws_stress.js
│   └── README.md
└── ...
```

---

## Roadmap (Optional)
- Add Redis pub/sub for multi-instance WebSocket fanout
- Add message persistence + search (Postgres)
- Add rate limiting + abuse protection
- Add CI workflow to run smoke tests on PRs

---

## License
Add your license here (MIT/Apache-2.0/etc). If you haven’t decided yet, you can add:
- `MIT` (simple, permissive), or
- keep “All Rights Reserved” until you choose.
