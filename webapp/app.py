"""
NAV4RAIL BT Generator — FastAPI Web Application.

Usage:
    uvicorn app:app --host 0.0.0.0 --port 8000
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from inference import Nav4RailGenerator, TEST_MISSIONS, validate_bt

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "nav4rail-mistral-7b-q4_k_m.gguf"
HISTORY_PATH = APP_DIR / "history.json"

app = FastAPI(title="NAV4RAIL BT Generator")
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
templates = Jinja2Templates(directory=APP_DIR / "templates")

generator: Nav4RailGenerator | None = None
generation_lock = asyncio.Lock()

# ─── History persistence + SSE broadcast ─────────────────────────────────────

sse_clients: list[asyncio.Queue] = []
generation_start: float | None = None


def load_history() -> list[dict]:
    if HISTORY_PATH.exists():
        return json.loads(HISTORY_PATH.read_text())
    return []


def save_history(entries: list[dict]):
    HISTORY_PATH.write_text(json.dumps(entries, ensure_ascii=False, indent=2))


async def broadcast(event: str, data: dict | list):
    payload = json.dumps(data, ensure_ascii=False)
    for q in list(sse_clients):
        try:
            q.put_nowait((event, payload))
        except asyncio.QueueFull:
            pass


async def broadcast_history():
    await broadcast("history", load_history())


def append_history(entry: dict):
    entries = load_history()
    entries.insert(0, entry)
    save_history(entries)


# ─── Startup ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    global generator
    if MODEL_PATH.exists():
        print(f"[app] Loading model from {MODEL_PATH} ...")
        generator = Nav4RailGenerator(str(MODEL_PATH))
        print("[app] Model ready!")
    else:
        print(f"[app] Model not found at {MODEL_PATH}")
        print("[app] Place your .gguf file in webapp/models/ to enable generation.")


# ─── Request / Response models ───────────────────────────────────────────────

class GenerateRequest(BaseModel):
    mission: str
    use_grammar: bool = True


class ValidateRequest(BaseModel):
    xml: str


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/status")
async def status():
    return {"loaded": generator is not None and generator.loaded}


@app.get("/api/examples")
async def examples():
    return {"missions": TEST_MISSIONS}


@app.get("/api/history")
async def get_history():
    return load_history()


@app.get("/api/history/stream")
async def history_stream(request: Request):
    queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue(maxsize=32)
    sse_clients.append(queue)

    async def event_generator():
        try:
            # Send current generation status on connect
            if generation_start is not None:
                yield {
                    "event": "gen_status",
                    "data": json.dumps({
                        "status": "running",
                        "started_at": generation_start,
                    }),
                }
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event, data = await asyncio.wait_for(queue.get(), timeout=30)
                    yield {"event": event, "data": data}
                except asyncio.TimeoutError:
                    yield {"event": "ping", "data": ""}
        finally:
            sse_clients.remove(queue)

    return EventSourceResponse(event_generator())


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    if generator is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Model not loaded. Place .gguf in webapp/models/."},
        )

    if generation_lock.locked():
        return JSONResponse(
            status_code=429,
            content={"error": "Une generation est deja en cours. Veuillez patienter."},
        )

    global generation_start
    generation_start = time.time()
    await broadcast("gen_status", {
        "status": "running",
        "started_at": generation_start,
        "mission": req.mission,
    })

    try:
        async with generation_lock:
            result = await asyncio.to_thread(
                generator.generate, req.mission, req.use_grammar
            )
    finally:
        generation_start = None
        await broadcast("gen_status", {"status": "idle"})

    append_history({
        "mode": "generate",
        "mission": req.mission,
        "xml": result.get("xml", ""),
        "score": round(result.get("score", 0), 2),
        "valid": result.get("valid", False),
        "time": datetime.now().strftime("%H:%M:%S"),
    })
    await broadcast_history()

    return result


@app.post("/api/validate")
async def validate(req: ValidateRequest):
    vr = validate_bt(req.xml)

    append_history({
        "mode": "validate",
        "mission": None,
        "xml": req.xml,
        "score": round(vr.score, 2),
        "valid": vr.valid,
        "time": datetime.now().strftime("%H:%M:%S"),
    })
    await broadcast_history()

    return {
        "valid": vr.valid,
        "score": round(vr.score, 2),
        "errors": vr.errors,
        "warnings": vr.warnings,
        "summary": vr.summary(),
    }
