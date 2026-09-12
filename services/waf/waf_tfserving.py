"""
Lightweight WAF that calls TF-Serving (or mock) and has conservative rule-based fallback.

Explainable WAF:
 - /predict: hybrid rule + model
 - /explain: tokens + weights (0..1) + risk_score (0..100) + reasons + rule_hits
"""

import os
import json
import re
from datetime import datetime
from typing import Optional, Tuple, List

import numpy as np
import requests
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

from explain_utils import (
    simple_tokenize,
    token_weights,
    build_reasons,
    calc_risk_score,
    top_tokens,
    match_rules as explain_match_rules,
    pick_primary_rule as explain_pick_primary_rule,
)

# ---------- config ----------
TF_HOST = os.environ.get("TF_SERVING_HOST", "tfserving")
TF_PORT = os.environ.get("TF_SERVING_PORT", "8501")
TF_MODEL = os.environ.get("TF_MODEL_NAME", "sqli")
THRESHOLD = float(os.environ.get("THRESHOLD", 0.5))
MAXLEN = int(os.environ.get("MAXLEN", 50))
VOCAB_PATH = os.environ.get("TOKENIZER_PATH", "models/tokenizer_vocab.json")
WAF_LOG_PATH = os.environ.get("WAF_LOG_PATH", "/app/waf_blocked.log")
LOG_TAIL_LINES = int(os.environ.get("LOG_TAIL_LINES", 50))

# ---------- SOC config ----------
SOC_ENABLED = os.environ.get("SOC_ENABLED", "0") == "1"
SOC_URL = os.environ.get("SOC_URL", "http://soc:5210/ingest")
SOC_TIMEOUT = float(os.environ.get("SOC_TIMEOUT", 1.0))

# ---------- RL config ----------
RL_THRESHOLD_MIN = 0.5
RL_THRESHOLD_MAX = 0.99

# ---------- fastapi ----------
app = FastAPI(title="SQLi WAF - improved demo (Explainable)")

# ---------- pad sequences (no TF) ----------
def pad_sequences_simple(seqs, maxlen, padding="post", truncating="post", value=0):
    out = np.full((len(seqs), maxlen), fill_value=value, dtype=int)
    for i, s in enumerate(seqs):
        if not s:
            continue
        if len(s) <= maxlen:
            if padding == "post":
                out[i, : len(s)] = s
            else:
                out[i, -len(s) :] = s
        else:
            trimmed = s[:maxlen] if truncating == "post" else s[-maxlen:]
            if padding == "post":
                out[i] = trimmed
            else:
                out[i, -maxlen:] = trimmed
    return out

# ---------- load tokenizer vocab ----------
if not os.path.exists(VOCAB_PATH):
    raise RuntimeError(
        f"tokenizer vocab not found at {VOCAB_PATH}. "
        f"Run export_tokenizer_vocab.py on host first."
    )

with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab = json.load(f)

WORD_INDEX = {k: v for k, v in vocab.get("word_index", {}).items()}
OOV_TOKEN = vocab.get("oov_token", None)
OOV_INDEX = WORD_INDEX.get(OOV_TOKEN) if OOV_TOKEN else None

# ---------- tokenizer for model ----------
_token_split_re = re.compile(r"\w+|[^\s\w]", re.UNICODE)

def simple_text_to_word_sequence(text, lower=True):
    if text is None:
        return []
    if lower:
        text = text.lower()
    return _token_split_re.findall(text)

def texts_to_sequences(texts):
    seqs = []
    for t in texts:
        toks = simple_text_to_word_sequence(t, lower=True)
        seq = []
        for tok in toks:
            if tok in WORD_INDEX:
                seq.append(WORD_INDEX[tok])
            else:
                seq.append(OOV_INDEX if OOV_INDEX is not None else 0)
        seqs.append(seq)
    return seqs

# ---------- WAF rules for /predict ----------
RULES: List[Tuple[str, re.Pattern]] = [
    ("tautology", re.compile(r"(?i)\b(or|and)\b\s+\d+\s*=\s*\d+")),
    ("union_select", re.compile(r"(?i)\bunion\b[\s\S]*\bselect\b")),
    ("sql_comment", re.compile(r"(--|/\*|#)")),
    ("stacked_danger", re.compile(r";\s*(?:drop|delete|insert|update|alter|create)\b", re.I)),
    ("time_function", re.compile(r"(?i)\b(sleep|benchmark|pg_sleep|waitfor)\b\s*\(")),
    ("long_hex", re.compile(r"0x[0-9a-fA-F]{6,}")),
    ("information_schema_recon", re.compile(r"(?i)\binformation_schema\b")),
]

def match_rules_predict(query: str) -> List[str]:
    hits = []
    for name, rx in RULES:
        if rx.search(query):
            hits.append(name)
    return hits

# ---------- heuristic: obviously benign queries ----------
def is_obviously_benign(query: str) -> bool:
    q = " ".join(query.lower().split())

    if q.startswith("insert into") and " union " not in q and " select " not in q and " or " not in q and "--" not in q and "/*" not in q:
        return True
    if q.startswith("update ") and " union " not in q and " select " not in q and " or " not in q and "--" not in q and "/*" not in q:
        return True
    if q.startswith("delete from") and " union " not in q and " select " not in q and " or " not in q and "--" not in q and "/*" not in q:
        return True
    if q.startswith("select count(") and " union " not in q and " or " not in q and "--" not in q and "/*" not in q:
        return True
    if q.startswith("select ") and " from " in q and " where " in q and " union " not in q and " or " not in q and "--" not in q and "/*" not in q:
        return True

    return False

# ---------- logging ----------
def waf_log_block(query: str, reason: str):
    ts = datetime.utcnow().isoformat() + "Z"
    line = f"{ts} WARNING BLOCK({reason}) query={query!r}\n"
    try:
        with open(WAF_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        print("WAF log write failed:", WAF_LOG_PATH)
    print(line.strip())

def tail_file(path: str, lines: int = 50) -> str:
    try:
        with open(path, "rb") as f:
            data = f.read()
        text = data.decode("utf-8", errors="replace")
        arr = text.splitlines()
        return "\n".join(arr[-lines:])
    except FileNotFoundError:
        return ""
    except Exception as e:
        return f"Error reading log: {e}"

def send_soc_event(request: Request, query: str, decision: str, proba: float, rule_hits: list):
    if not SOC_ENABLED:
        return

    try:
        event = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "src_ip": request.client.host if request.client else "unknown",
            "endpoint": str(request.url.path),
            "query": query,
            "decision": decision,
            "prediction": "SQLi Attack" if decision == "BLOCK" else "Benign",
            "probability": float(proba),
            "threshold": THRESHOLD,
            "risk_score": int(proba * 100),
            "rule_hits": rule_hits,
            "reasons": rule_hits if rule_hits else ["benign"],
        }
        requests.post(SOC_URL, json=event, timeout=SOC_TIMEOUT)
    except Exception as e:
        print("[SOC ERROR]", e)

# ---------- RL ----------
def adjust_threshold(correct: bool):
    global THRESHOLD

    if correct:
        THRESHOLD -= 0.005
    else:
        THRESHOLD += 0.01

    THRESHOLD = max(RL_THRESHOLD_MIN, min(RL_THRESHOLD_MAX, THRESHOLD))
    print(f"[RL] Updated threshold: {THRESHOLD}")

class QueryIn(BaseModel):
    query: str

class RLUpdateIn(BaseModel):
    correct: bool = True

@app.get("/")
def root():
    return {
        "message": "SQLi WAF API running. Use POST /predict, POST /explain, or GET /demo",
        "threshold": THRESHOLD,
        "maxlen": MAXLEN,
    }

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/demo")
def demo():
    demo_path = os.path.join(os.getcwd(), "index.html")
    if os.path.exists(demo_path):
        return FileResponse(demo_path, media_type="text/html")
    return JSONResponse({"detail": "demo not found"}, status_code=404)

@app.get("/logs")
def logs():
    txt = tail_file(WAF_LOG_PATH, LOG_TAIL_LINES)
    return PlainTextResponse(txt or "(no logs)")

@app.post("/rl/update")
def rl_update(data: RLUpdateIn):
    adjust_threshold(data.correct)
    return {"status": "updated", "threshold": THRESHOLD}

# ---------- helper: call TF-Serving ----------
def call_model_proba(query_text: str) -> Tuple[float, Optional[str]]:
    seq = texts_to_sequences([query_text])
    X = pad_sequences_simple(seq, maxlen=MAXLEN, padding="post", truncating="post", value=0)

    payload = {"instances": X.tolist()}
    url = f"http://{TF_HOST}:{TF_PORT}/v1/models/{TF_MODEL}:predict"

    try:
        resp = requests.post(url, json=payload, timeout=6)
        resp.raise_for_status()
        data = resp.json()
        proba = float(data["predictions"][0][0])
        return proba, None
    except Exception as e:
        return 0.0, str(e)

# ---------- /predict ----------
@app.post("/predict")
def predict(qin: QueryIn, request: Request):
    query_text = (qin.query or "").strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Empty query")

    if is_obviously_benign(query_text):
        print(f"[RULE SAFE] skip model, mark Benign: {query_text!r}")
        send_soc_event(request, query_text, "ALLOW", 0.0, ["rule_allow"])
        return {
            "query": query_text,
            "prediction": "Benign",
            "probability": 0.0,
            "threshold": THRESHOLD,
            "explanation": "rule_allow",
        }

    proba, model_error = call_model_proba(query_text)

    if model_error is None and proba >= THRESHOLD:
        waf_log_block(query_text, "model")
        send_soc_event(request, query_text, "BLOCK", proba, ["model"])
        return {
            "query": query_text,
            "prediction": "SQLi Attack",
            "probability": float(proba),
            "threshold": THRESHOLD,
            "explanation": "model",
        }

    rule_hits = match_rules_predict(query_text)
    if rule_hits:
        waf_log_block(query_text, ",".join(rule_hits))
        send_soc_event(request, query_text, "BLOCK", proba, rule_hits)
        return {
            "query": query_text,
            "prediction": "SQLi Attack",
            "probability": float(proba),
            "threshold": THRESHOLD,
            "explanation": f"rule:{','.join(rule_hits)}",
        }

    explanation = "model" if model_error is None else "no_model"
    send_soc_event(request, query_text, "ALLOW", proba, [])
    return {
        "query": query_text,
        "prediction": "Benign",
        "probability": float(proba),
        "threshold": THRESHOLD,
        "explanation": explanation,
    }

# ---------- /explain ----------
@app.post("/explain")
def explain(qin: QueryIn, request: Request):
    query_text = (qin.query or "").strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Empty query")

    if is_obviously_benign(query_text):
        tokens = simple_tokenize(query_text)
        benign_weights = [0.05 for _ in tokens]
        return {
            "query": query_text,
            "prediction": "Benign",
            "probability": 0.0,
            "threshold": float(THRESHOLD),
            "risk_score": 0,
            "tokens": tokens,
            "weights": benign_weights,
            "reasons": ["RULE:ALLOW_BENIGN"],
            "top_tokens": top_tokens(tokens, benign_weights, k=5),
            "rule_hits": [],
            "primary_rule": None,
            "model_status": "skipped",
        }

    proba, model_error = call_model_proba(query_text)
    rule_hits = explain_match_rules(query_text)
    primary_rule = explain_pick_primary_rule(rule_hits)
    tokens = simple_tokenize(query_text)
    weights = token_weights(tokens)
    reasons = build_reasons(query_text, rule_hits, proba, THRESHOLD)
    risk_score = calc_risk_score(proba, rule_hits)
    is_attack = (proba >= THRESHOLD) or (len(rule_hits) > 0)
    prediction = "SQLi Attack" if is_attack else "Benign"

    return {
        "query": query_text,
        "prediction": prediction,
        "probability": float(proba),
        "threshold": float(THRESHOLD),
        "risk_score": int(risk_score),
        "tokens": tokens,
        "weights": [float(x) for x in weights],
        "reasons": reasons,
        "top_tokens": top_tokens(tokens, weights, k=5),
        "rule_hits": rule_hits,
        "primary_rule": primary_rule,
        "model_status": "ok" if model_error is None else "no_model",
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("waf_tfserving:app", host="0.0.0.0", port=5110)