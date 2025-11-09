# start.py
import os, uvicorn, time, traceback
port = int(os.environ.get("PORT", "8000"))
print(f"STARTUP: Using PORT={port}", flush=True)
try:
    uvicorn.run("app:app", host="0.0.0.0", port=port, access_log=True, log_level="debug")
except Exception:
    print("STARTUP EXCEPTION:", flush=True)
    traceback.print_exc()
    time.sleep(10)
    raise
