import os
import uvicorn
from app import app  # Make sure app.py defines `app = FastAPI()`

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Railway sets PORT dynamically
    uvicorn.run(app, host="0.0.0.0", port=port)
