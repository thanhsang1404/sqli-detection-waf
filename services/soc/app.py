from datetime import datetime, timezone
from typing import List, Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from db import init_db, insert_alert, list_alerts, stats, update_feedback

app = FastAPI(title="MiniSOC - SQLi Alerts")

WAF_RL_URL = "http://waf:5110/rl/update"


class AlertEvent(BaseModel):
    ts: Optional[str] = None
    src_ip: Optional[str] = None
    endpoint: Optional[str] = None
    decision: str = Field(default="BLOCK")
    prediction: Optional[str] = None
    probability: Optional[float] = 0.0
    threshold: Optional[float] = 0.0
    risk_score: Optional[int] = 0
    reasons: Optional[List[str]] = None
    rule_hits: Optional[List[str]] = None
    query: Optional[str] = None


@app.on_event("startup")
def startup():
    init_db()


@app.get("/healthz")
def health():
    return {"status": "ok"}


@app.get("/")
def ui():
    return FileResponse("ui.html")


@app.post("/ingest")
def ingest(e: AlertEvent):
    payload = e.dict()

    if not payload.get("ts"):
        payload["ts"] = datetime.now(timezone.utc).isoformat()

    if not payload.get("query"):
        raise HTTPException(status_code=400, detail="Missing query")

    new_id = insert_alert(payload)
    print("[SOC] stored:", payload["query"])

    return {"status": "ok", "id": new_id}


@app.get("/alerts")
def alerts(limit: int = 50):
    return {"items": list_alerts(limit)}


@app.get("/stats")
def get_stats():
    return stats()


@app.post("/feedback")
def feedback(alert_id: int, correct: bool):
    update_feedback(alert_id, correct)
    print(f"[RL] Feedback received: id={alert_id}, correct={correct}")

    try:
        requests.post(
            WAF_RL_URL,
            json={"correct": correct},
            timeout=1
        )
    except Exception as e:
        print("[RL ERROR] Cannot reach WAF:", e)

    return {"status": "updated"}