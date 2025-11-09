# start.py
import os
import uvicorn

port = int(os.environ.get("PORT", "8000"))
print(f"STARTUP: Using PORT={port}", flush=True)
uvicorn.run("app:app", host="0.0.0.0", port=port)
