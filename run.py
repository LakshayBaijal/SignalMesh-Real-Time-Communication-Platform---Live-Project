# run.py — safe startup wrapper that prints full import errors to logs
import os
import sys
import traceback

# Try to import the application (this will surface import-time errors)
try:
    # server.py should define `app = FastAPI()` at top-level
    from server import app
except Exception as e:
    print("=== ERROR: failed to import server.py ===", file=sys.stderr)
    traceback.print_exc()
    # make sure Render logs show something obvious and exit non-zero
    sys.exit(1)

# If import succeeded, start Uvicorn here (so we control startup)
try:
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    print(f"Starting uvicorn via run.py on port {port}")
    # Use uvicorn.run(app, ...) so logs appear in Render console
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
except Exception:
    print("=== ERROR: failed to start uvicorn ===", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)