from fastapi import FastAPI
import asyncio
import traceback
import sys
import prototype9  # Your model logic lives here

app = FastAPI()

models_ready = False
_analyze_frame = None

@app.on_event("startup")
async def startup_event():
    global models_ready, _analyze_frame
    try:
        print("Model loader: initializing real models", flush=True)
        await asyncio.to_thread(prototype9.init_models)
        _analyze_frame = prototype9.analyze_frame
        print("Model loader: finished", flush=True)
    except Exception:
        print("Model loader exception:", file=sys.stderr, flush=True)
        traceback.print_exc()
    finally:
        models_ready = True
        print(f"Model loader: models_ready={models_ready}", flush=True)

@app.get("/health")
def health_check():
    return {"models_ready": models_ready}
