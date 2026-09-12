"""
Lightweight mock of TF-Serving for local testing.
- POST /v1/models/{model_name}:predict  (body: {"instances": [...]})
- GET  /v1/models/{model_name}:predict?q=...
- Health check: GET /healthz

+ Logging:
  - LOG_DIR (env) default: ./logs
  - mock_requests.log: request + predictions
  - mock_errors.log: exceptions
"""

from fastapi import FastAPI, Path, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Any, Optional
import numpy as np
import logging
import os
import uvicorn
from datetime import datetime


# -------------------------
# Logging setup
# -------------------------
LOG_DIR = os.environ.get("LOG_DIR", "./logs")
os.makedirs(LOG_DIR, exist_ok=True)

req_logger = logging.getLogger("mock_tfserving.requests")
err_logger = logging.getLogger("mock_tfserving.errors")
req_logger.setLevel(logging.INFO)
err_logger.setLevel(logging.INFO)

# Avoid duplicate handlers if reloaded
if not req_logger.handlers:
    req_fh = logging.FileHandler(os.path.join(LOG_DIR, "mock_requests.log"), encoding="utf-8")
    req_fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    req_logger.addHandler(req_fh)

if not err_logger.handlers:
    err_fh = logging.FileHandler(os.path.join(LOG_DIR, "mock_errors.log"), encoding="utf-8")
    err_fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    err_logger.addHandler(err_fh)

# Also print to stdout (Docker logs)
stream = logging.StreamHandler()
stream.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
req_logger.addHandler(stream)
err_logger.addHandler(stream)


# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="Mock TF-Serving (light)")

class PredictIn(BaseModel):
    instances: List[Any]

@app.on_event("startup")
async def startup_event():
    req_logger.info("Mock TF-Serving starting up... LOG_DIR=%s", LOG_DIR)

@app.get("/healthz")
async def healthz():
    return JSONResponse({"status": "ok"})


@app.post("/v1/models/{model_name}:predict")
async def predict_post(model_name: str = Path(...), body: PredictIn = None, request: Request = None):
    """
    Accepts {"instances": [...]} and returns {"predictions": [[prob], ...]}

    Heuristic logic (demo only):
      - empty / invalid -> 0.01
      - large arrays or large sums -> 0.99
      - otherwise: derived small probability from numeric summary
    """
    if body is None or body.instances is None:
        raise HTTPException(status_code=400, detail="Missing 'instances' in body")

    instances = body.instances
    out = []

    for inst in instances:
        try:
            arr = np.array(inst, dtype=float).flatten()
            if arr.size == 0:
                prob = 0.01
            elif arr.size > 50 or arr.sum() > 100:
                prob = 0.99
            else:
                s = float(np.abs(arr).sum())
                prob = float(min(0.9, (s % 10) / 10.0 + 0.05))
            out.append([round(float(prob), 6)])
        except Exception:
            # non-numeric fallback
            try:
                length = len(inst) if hasattr(inst, "__len__") else 1
                if length > 100:
                    prob = 0.95
                elif length > 20:
                    prob = 0.7
                else:
                    prob = 0.05
                out.append([round(float(prob), 6)])
            except Exception:
                out.append([0.01])

    # Log request + predictions
    client_ip = request.client.host if request and request.client else "unknown"
    req_logger.info(
        "POST predict model=%s client=%s instances=%d preds=%s",
        model_name, client_ip, len(instances), out
    )

    return {"predictions": out}


@app.get("/v1/models/{model_name}:predict")
async def predict_get(model_name: str = Path(...), q: Optional[str] = None, request: Request = None):
    """
    Simple GET fallback so browser/curl doesn't 405.
    Example: GET /v1/models/sqli:predict?q=1+OR+1=1
    """
    if q is None:
        preds = [[0.01]]
        req_logger.info("GET predict model=%s q=None -> %s", model_name, preds)
        return {"predictions": preds}

    text = str(q).lower()
    if "1=1" in text or " or " in text or "--" in text:
        preds = [[0.95]]
    else:
        preds = [[0.05]]

    client_ip = request.client.host if request and request.client else "unknown"
    req_logger.info("GET predict model=%s client=%s q=%r -> %s", model_name, client_ip, q, preds)
    return {"predictions": preds}


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    err_logger.exception("Unhandled error path=%s err=%s", str(request.url.path), exc)
    return JSONResponse(status_code=500, content={"detail": "internal error"})


if __name__ == "__main__":
    uvicorn.run("mock_tfserving:app", host="0.0.0.0", port=8501, log_level="info")
