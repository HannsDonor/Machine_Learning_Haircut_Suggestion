# Background loader
async def load_models():
    global models_ready, _analyze_frame
    try:
        import prototype9
        print("Model loader: initializing real models", flush=True)
        await asyncio.to_thread(prototype9.init_models)  # load your real models
        _analyze_frame = prototype9.analyze_frame
        print("Model loader: finished", flush=True)
    except Exception:
        print("Model loader exception:", file=sys.stderr, flush=True)
        traceback.print_exc()
    finally:
        models_ready = True
        print(f"Model loader: models_ready={models_ready}", flush=True)
