import os
import sqlite3
from typing import Any, Dict, List

DB_PATH = os.environ.get("SOC_DB_PATH", "/app/data/alerts.db")

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS alerts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  src_ip TEXT,
  endpoint TEXT,
  decision TEXT,
  prediction TEXT,
  probability REAL,
  threshold REAL,
  risk_score INTEGER,
  reasons TEXT,
  rule_hits TEXT,
  query TEXT,
  feedback INTEGER
);
"""

def _connect():
  os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
  conn = sqlite3.connect(DB_PATH, check_same_thread=False)
  conn.row_factory = sqlite3.Row
  return conn

_conn = None

def _get_conn():
  global _conn
  if _conn is None:
    _conn = _connect()
  return _conn

def init_db():
  conn = _get_conn()
  conn.execute(CREATE_TABLE_SQL)
  conn.commit()

  cols = [r["name"] for r in conn.execute("PRAGMA table_info(alerts)").fetchall()]
  if "feedback" not in cols:
    conn.execute("ALTER TABLE alerts ADD COLUMN feedback INTEGER")
    conn.commit()

# =========================
# ✅ FIX CHUẨN Ở ĐÂY
# =========================
def get_all_alerts(limit: int = 50):
  init_db()
  conn = _get_conn()

  limit = max(1, min(int(limit or 50), 500))

  rows = conn.execute(
    "SELECT * FROM alerts ORDER BY id DESC LIMIT ?",
    (limit,),
  ).fetchall()

  return {
    "items": [dict(r) for r in rows]
  }

def insert_alert(e: Dict[str, Any]) -> int:
  init_db()
  conn = _get_conn()

  reasons = e.get("reasons", "")
  if isinstance(reasons, list):
    reasons = ",".join([str(x) for x in reasons])

  rule_hits = e.get("rule_hits", "")
  if isinstance(rule_hits, list):
    rule_hits = ",".join([str(x) for x in rule_hits])

  cur = conn.execute(
    """
    INSERT INTO alerts (
      ts, src_ip, endpoint, decision, prediction, probability, threshold,
      risk_score, reasons, rule_hits, query, feedback
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
    (
      str(e.get("ts", "")),
      str(e.get("src_ip", "")),
      str(e.get("endpoint", "")),
      str(e.get("decision", "")),
      str(e.get("prediction", "")),
      float(e.get("probability", 0.0) or 0.0),
      float(e.get("threshold", 0.0) or 0.0),
      int(e.get("risk_score", 0) or 0),
      str(reasons),
      str(rule_hits),
      str(e.get("query", "")),
      None,
    ),
  )
  conn.commit()
  return int(cur.lastrowid)

def update_feedback(alert_id: int, correct: bool):
  init_db()
  conn = _get_conn()
  conn.execute(
    "UPDATE alerts SET feedback=? WHERE id=?",
    (1 if correct else 0, alert_id),
  )
  conn.commit()

def list_alerts(limit: int = 50) -> List[Dict[str, Any]]:
  init_db()
  conn = _get_conn()

  limit = max(1, min(int(limit or 50), 500))

  rows = conn.execute(
    "SELECT * FROM alerts ORDER BY id DESC LIMIT ?",
    (limit,),
  ).fetchall()

  return [dict(r) for r in rows]

def stats() -> Dict[str, Any]:
  init_db()
  conn = _get_conn()

  total = int(conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0])
  blocked = int(conn.execute("SELECT COUNT(*) FROM alerts WHERE decision='BLOCK'").fetchone()[0])
  allowed = int(conn.execute("SELECT COUNT(*) FROM alerts WHERE decision='ALLOW'").fetchone()[0])
  feedback_correct = int(conn.execute("SELECT COUNT(*) FROM alerts WHERE feedback=1").fetchone()[0])
  feedback_wrong = int(conn.execute("SELECT COUNT(*) FROM alerts WHERE feedback=0").fetchone()[0])

  return {
    "total": total,
    "blocked": blocked,
    "allowed": allowed,
    "feedback_correct": feedback_correct,
    "feedback_wrong": feedback_wrong,
    "db_path": DB_PATH,
  }