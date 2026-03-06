"""
NAV4RAIL BT Generator — FastAPI Web Application.

Usage:
    uvicorn app:app --host 0.0.0.0 --port 8000
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from inference import Nav4RailGenerator, TEST_MISSIONS, validate_bt
from inference_nav2 import TEST_MISSIONS_NAV2, build_nav2_generator_from_env
from nav2_pipeline import build_xml_from_steps, load_nav2_catalog, parse_steps_payload
from ros_nav2_client import RosNav2Client

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "nav4rail-mistral-7b-q4_k_m.gguf"

app = FastAPI(title="NAV4RAIL BT Generator")
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
templates = Jinja2Templates(directory=APP_DIR / "templates")

generator: Nav4RailGenerator | None = None
nav2_generator = build_nav2_generator_from_env()
ros_nav2_client = RosNav2Client()
nav2_catalog = load_nav2_catalog()


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


class Nav2GenerateRequest(BaseModel):
    mission: str
    constrained: str = "jsonschema"
    max_new_tokens: int = 256
    temperature: float = 0.0
    write_run: bool = True
    strict_attrs: bool = True
    strict_blackboard: bool = True


class Nav2StepsRequest(BaseModel):
    steps_json: str
    strict_attrs: bool = True
    strict_blackboard: bool = True
    write_run: bool = False


class Nav2ExecuteRequest(BaseModel):
    xml: Optional[str] = None
    filename: Optional[str] = None
    goal_pose: Optional[str] = None
    goal_name: Optional[str] = None
    initial_pose: Optional[str] = "0.0,0.0,0.0"
    allow_invalid: bool = False
    start_stack_if_needed: bool = True
    restart_navigation: bool = True


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/status")
async def status():
    return {"loaded": generator is not None and generator.loaded}


@app.get("/api/modes")
async def modes():
    return {
        "default_mode": "legacy",
        "modes": [
            {"id": "legacy", "label": "NAV4RAIL XML direct"},
            {"id": "nav2", "label": "ROS2 Nav2 steps JSON"},
        ],
    }


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


@app.get("/api/nav2/status")
async def nav2_status():
    status_payload = nav2_generator.status()
    return {
        "loaded": status_payload["loaded"],
        "configured": status_payload["configured"],
        "model_key": status_payload["model_key"],
        "adapter_dir": status_payload["adapter_dir"],
        "load_error": status_payload["load_error"],
    }


@app.get("/api/nav2/examples")
async def nav2_examples():
    return {"missions": TEST_MISSIONS_NAV2}


@app.post("/api/nav2/generate")
async def nav2_generate(req: Nav2GenerateRequest):
    try:
        result = await asyncio.to_thread(
            nav2_generator.generate,
            req.mission,
            constrained=req.constrained,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            write_run=req.write_run,
            strict_attrs=req.strict_attrs,
            strict_blackboard=req.strict_blackboard,
        )
        return result
    except Exception as exc:
        return JSONResponse(status_code=503, content={"error": str(exc)})


@app.post("/api/nav2/validate/steps")
async def nav2_validate_steps(req: Nav2StepsRequest):
    parsed_payload = parse_steps_payload(req.steps_json, catalog=nav2_catalog)
    return {
        "valid": parsed_payload["ok"],
        "steps": parsed_payload["steps"],
        "steps_json": parsed_payload["steps_json"],
        "parse_error": parsed_payload["error_message"],
        "parse_error_counters": parsed_payload["error_counters"],
        "errors": parsed_payload["errors"],
        "summary": "Steps JSON valid" if parsed_payload["ok"] else (parsed_payload["error_message"] or "Steps JSON invalid"),
    }


@app.post("/api/nav2/steps-to-xml")
async def nav2_steps_to_xml(req: Nav2StepsRequest):
    parsed_payload = parse_steps_payload(req.steps_json, catalog=nav2_catalog)
    if not parsed_payload["ok"] or not parsed_payload["steps"]:
        return JSONResponse(
            status_code=422,
            content={
                "error": parsed_payload["error_message"] or "Steps JSON invalid",
                "errors": parsed_payload["errors"],
                "parse_error_counters": parsed_payload["error_counters"],
            },
        )

    xml_payload = build_xml_from_steps(
        parsed_payload["steps"],
        catalog=nav2_catalog,
        strict_attrs=req.strict_attrs,
        strict_blackboard=req.strict_blackboard,
    )
    xml_payload["steps"] = parsed_payload["steps"]
    xml_payload["steps_json"] = parsed_payload["steps_json"]
    return xml_payload


@app.post("/api/nav2/validate/xml")
async def nav2_validate_xml(req: ValidateRequest):
    import tempfile

    from finetune_Nav2.eval.bt_validation import validate_bt_xml

    with tempfile.NamedTemporaryFile("w+", suffix=".xml", delete=False, encoding="utf-8") as tmp:
        tmp.write(req.xml)
        tmp.flush()
        tmp_path = Path(tmp.name)
    try:
        report = validate_bt_xml(xml_path=tmp_path, strict_attrs=True, strict_blackboard=True)
    finally:
        tmp_path.unlink(missing_ok=True)

    errors = []
    warnings = []
    for issue in report.get("issues", []) or []:
        line = f"[{issue.get('code', 'unknown')}] {issue.get('message', '')}".strip()
        if str(issue.get("level", "error")).lower() == "warning":
            warnings.append(line)
        else:
            errors.append(line)

    return {
        "xml": req.xml,
        "valid": bool(report.get("ok")),
        "score": 1.0 if bool(report.get("ok")) else 0.0,
        "errors": errors,
        "warnings": warnings,
        "summary": "Strict validator passed" if bool(report.get("ok")) else "Strict validator failed",
        "validation_report": report,
        "steps_json": "",
        "steps": [],
    }


@app.post("/api/nav2/transfer")
async def nav2_transfer(req: Nav2ExecuteRequest):
    if not req.xml:
        return JSONResponse(status_code=422, content={"error": "xml is required for transfer"})
    try:
        result = await asyncio.to_thread(
            ros_nav2_client.upload_bt,
            xml=req.xml,
            filename=req.filename,
        )
        return result
    except Exception as exc:
        return JSONResponse(status_code=503, content={"error": str(exc)})


@app.post("/api/nav2/execute")
async def nav2_execute(req: Nav2ExecuteRequest):
    try:
        result = await asyncio.to_thread(
            ros_nav2_client.execute_bt,
            xml=req.xml,
            filename=req.filename,
            goal_pose=req.goal_pose,
            goal_name=req.goal_name,
            initial_pose=req.initial_pose,
            allow_invalid=req.allow_invalid,
            start_stack_if_needed=req.start_stack_if_needed,
            restart_navigation=req.restart_navigation,
        )
        result["requested_at"] = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        return result
    except Exception as exc:
        return JSONResponse(status_code=503, content={"error": str(exc)})
