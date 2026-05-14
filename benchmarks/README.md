# SignalMesh Benchmark Guide (k6)

This folder contains reproducible benchmark scripts for your deployed SignalMesh app.

## Files
- `http_load.js`: lightly records HTTP latency/error rate for `/ping` and `/login` on a free-tier server
- `ws_load.js`: lightly records authenticated WebSocket connection time and message round-trip latency

## 1) Install k6 (Windows PowerShell)
Use one of these:

```powershell
winget install k6.k6
```

or

```powershell
choco install k6
```

## 2) Move to project root
```powershell
Set-Location "d:\Projects\Perfectly Working SignalMesh-Real-Time-Communication-Platform---Live-Project-main_2\SignalMesh-Real-Time-Communication-Platform---Live-Project-main"
```

## 3) Set target URL and credentials for this terminal session
```powershell
$env:BASE_URL = "https://signal-mesh.onrender.com"
$env:BENCH_USERNAME = "bench_login_user"
$env:BENCH_PASSWORD = "BenchPass123!"
$env:WS_HOLD_SECONDS = "15"
$env:MESSAGE_INTERVAL_SECONDS = "3"
```

## 4) Run HTTP benchmark (latency + login throughput)
```powershell
k6 run .\benchmarks\http_load.js *> .\benchmarks\run_http.log
```

## 5) Run WebSocket benchmark (concurrency + round-trip)
```powershell
k6 run .\benchmarks\ws_load.js *> .\benchmarks\run_ws.log
```

## 6) Optional stress run (higher load)
```powershell
$env:CONNECTED_USERS = "100"
k6 run .\benchmarks\ws_stress.js *> .\benchmarks\run_ws_stress.log
```

## 7) Metrics to report in resume
Use only metrics directly visible in k6 summary/output:
- HTTP: `http_req_duration` p50/p95/p99, `http_reqs`, `http_req_failed`
- WebSocket load: `ws_connect_time_ms`, `ws_roundtrip_ms`, `ws_messages_sent`, `ws_messages_received`
- WebSocket stress: `websocket connected`, `ws_connections`, `ws_connection_checks`

## 8) Reproducibility rule
Run each scenario **3 times** and keep the cleanest stable numbers from the free-server runs.

## 9) Example claim template (fill with your real numbers)
- "Benchmarked deployed FastAPI/WebSocket service with k6 on a free-tier server; recorded HTTP and WebSocket latency, connection time, and message throughput from stable low-load runs."

## Notes
- `ws_load.js` uses one lightweight test account during `setup()`.
- If the free server is slow, keep `USERS=1` and rerun later instead of increasing the load.
- Keep benchmark credentials non-sensitive and dedicated to testing.


### Commands
```br
k6 run .\benchmarks\ws_load.js  --out json=.\benchmarks\ws_load.json  *> .\benchmarks\ws_load.log
```
```br
k6 run .\benchmarks\ws_stress.js --out json=.\benchmarks\ws_stress.json *> .\benchmarks\ws_stress.log
```
```br
k6 run .\benchmarks\http_load.js  --out json=.\benchmarks\http_load.json  *> .\benchmarks\http_load.log
```
```br
k6 run .\benchmarks\http_stress.js --out json=.\benchmarks\http_stress.json *> .\benchmarks\http_stress.log
```