from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import pymysql
import requests
import os
from typing import Optional

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# SERVE UI
# ---------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    with open("web.html", "r", encoding="utf-8") as f:
        return f.read()

# ---------------------------
# PROXY SOC — fix: browser không gọi được localhost:5210 trong Docker
# ---------------------------
SOC_URL = os.getenv("SOC_URL", "http://soc:5210")

@app.get("/soc/stats")
def soc_stats():
    try:
        res = requests.get(f"{SOC_URL}/stats", timeout=3)
        return res.json()
    except Exception as e:
        return {"total": 0, "blocked": 0, "allowed": 0, "error": str(e)}

@app.get("/soc/alerts")
def soc_alerts(limit: int = 50):
    try:
        res = requests.get(f"{SOC_URL}/alerts?limit={limit}", timeout=3)
        return res.json()
    except Exception as e:
        return {"items": [], "error": str(e)}

# ---------------------------
# MYSQL CONNECTION
# ---------------------------
def get_conn():
    return pymysql.connect(
        host=os.getenv("DB_HOST", "db"),
        user=os.getenv("DB_USER", "user"),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", "sqli_demo"),
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False
    )

# ---------------------------
# CALL WAF
# ---------------------------
WAF_URL = os.getenv("WAF_URL", "http://waf:5110")

def call_waf(query: str):
    try:
        res = requests.post(
            f"{WAF_URL}/predict",
            json={"query": query},
            timeout=3
        )
        return res.json()
    except Exception as e:
        print("[WAF ERROR]:", e)
        return {"prediction": "UNKNOWN", "probability": 0, "risk_score": 0}

# ---------------------------
# SEND SOC EVENT
# ---------------------------
def send_soc(endpoint: str, decision: str, query: str, waf_result: Optional[dict] = None):
    waf_result = waf_result or {}
    payload = {
        "src_ip": "web",
        "endpoint": endpoint,
        "decision": decision,
        "prediction": waf_result.get("prediction", "Benign" if decision == "ALLOW" else "SQLi Attack"),
        "probability": float(waf_result.get("probability", 0) or 0),
        "threshold": float(waf_result.get("threshold", 0.9) or 0.9),
        "risk_score": int(waf_result.get("risk_score", 0) or 0),
        "reasons": waf_result.get("reasons") or [],
        "rule_hits": waf_result.get("rule_hits") or [],
        "query": query,
    }
    try:
        requests.post(f"{SOC_URL}/ingest", json=payload, timeout=3)
    except Exception as e:
        print("[SOC ERROR]:", e)

# ---------------------------
# REGISTER
# ---------------------------
@app.post("/register")
def register(username: str = Form(...), password: str = Form(...)):
    query_check = f"INSERT INTO users(username, password) VALUES('{username}', '{password}')"
    waf = call_waf(query_check)

    if waf.get("prediction") == "SQLi Attack":
        send_soc("/register", "BLOCK", query_check, waf)
        return {"status": "BLOCKED", "reason": "SQL Injection detected", "query": query_check}

    conn = get_conn()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users(username, password) VALUES (%s, %s)", (username, password))
        conn.commit()
        send_soc("/register", "ALLOW", query_check, waf)
        return {"status": "REGISTERED", "query": query_check}
    except Exception as e:
        conn.rollback()
        return {"status": "ERROR", "error": str(e)}
    finally:
        conn.close()

# ---------------------------
# LOGIN
# ---------------------------
@app.post("/login")
def login(username: str = Form(...), password: str = Form(...)):
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
    waf = call_waf(query)

    if waf.get("prediction") == "SQLi Attack":
        send_soc("/login", "BLOCK", query, waf)
        return {"status": "BLOCKED", "reason": "SQL Injection detected", "query": query}

    conn = get_conn()
    c = conn.cursor()
    try:
        # Dùng prepared statement — tránh SQLi thật
        c.execute("SELECT * FROM users WHERE username=%s AND password=%s", (username, password))
        result = c.fetchone()
        send_soc("/login", "ALLOW", query, waf)
        if result:
            return {"status": "SUCCESS", "message": "Login successful", "query": query}
        else:
            return {"status": "FAIL", "message": "Invalid username or password", "query": query}
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}
    finally:
        conn.close()

# ---------------------------
# SEARCH
# ---------------------------
@app.get("/search")
def search(q: str):
    query = f"SELECT * FROM users WHERE username LIKE '%{q}%'"
    waf = call_waf(query)

    if waf.get("prediction") == "SQLi Attack":
        send_soc("/search", "BLOCK", query, waf)
        return {"status": "BLOCKED", "reason": "SQL Injection detected", "query": query}

    conn = get_conn()
    c = conn.cursor()
    try:
        # FIX: dùng prepared statement thay vì raw SQL
        c.execute("SELECT * FROM users WHERE username LIKE %s", (f"%{q}%",))
        result = c.fetchall()
        send_soc("/search", "ALLOW", query, waf)
        return {"status": "SUCCESS", "data": result, "query": query}
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}
    finally:
        conn.close()
