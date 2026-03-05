"""
NAV4RAIL BT Generator — FastAPI Web Application.

Usage:
    uvicorn app:app --host 0.0.0.0 --port 8000
"""

import asyncio
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from inference import Nav4RailGenerator, TEST_MISSIONS, validate_bt

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "nav4rail-mistral-7b-q4_k_m.gguf"

app = FastAPI(title="NAV4RAIL BT Generator")
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
templates = Jinja2Templates(directory=APP_DIR / "templates")

generator: Nav4RailGenerator | None = None


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


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    if generator is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Model not loaded. Place .gguf in webapp/models/."},
        )

    result = await asyncio.to_thread(
        generator.generate, req.mission, req.use_grammar
    )
    return result


@app.post("/api/validate")
async def validate(req: ValidateRequest):
    vr = validate_bt(req.xml)
    return {
        "valid": vr.valid,
        "score": round(vr.score, 2),
        "errors": vr.errors,
        "warnings": vr.warnings,
        "summary": vr.summary(),
    }
