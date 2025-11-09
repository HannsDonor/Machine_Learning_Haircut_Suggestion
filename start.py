# start.py
import os
import uvicorn

port = int(os.environ.get("PORT", "8000"))
print(f"STARTUP: Using PORT={port}", flush=True)

# run uvicorn with access log enabled so incoming HTTP requests are logged
uvicorn.run("app:app", host="0.0.0.0", port=port, access_log=True, log_level="info")
