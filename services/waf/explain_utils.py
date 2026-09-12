# explain_utils.py
import re
from typing import List, Dict, Optional

# =========================================================
# Tokenizer cho /explain
# - Ưu tiên token multi-char trước: "--", "/*", "*/"
# - Giữ ";" là token riêng
# - Bắt 0x... (hex) thành 1 token
# - Bắt "1=1" (và nói chung d+=d+) thành 1 token (không bị tách)
# - Cuối cùng mới đến \w+ và ký tự lẻ
# =========================================================
_TOKEN_RE = re.compile(
    r"(--|/\*|\*/|#|;|\b0x[0-9a-fA-F]{6,}\b|\b\d+\s*=\s*\d+\b|\w+|[^\s\w])",
    re.UNICODE,
)

# =========================================================
# Token nghi vấn (demo explain)
# Lưu ý: nếu bạn thấy SELECT/WHERE/UPDATE... bị đỏ cho benign nhiều quá
# thì nên hạ weight (xem phần token_weights) hoặc bỏ khỏi SUSPECT_TOKENS.
# =========================================================
SUSPECT_TOKENS = {
    "or", "and", "union", "select", "where",
    "--", "/*", "*/", "#", ";",
    "sleep", "benchmark", "pg_sleep", "waitfor",
    "xp_", "drop", "delete", "insert", "update", "alter", "create",
    "1=1", "'or", '"or', "'and", '"and'
}

RULE_CODE_MAP = {
    "tautology": "RULE:TAUTOLOGY",
    "union_select": "RULE:UNION_SELECT",
    "sql_comment": "RULE:COMMENT",
    "stacked_danger": "RULE:STACKED_DANGER",
    "time_function": "RULE:TIME_DELAY",
    "long_hex": "RULE:HEX_BLOB",
    "information_schema_recon": "RULE:INFO_SCHEMA_RECON",
}

# ---- RULE regex (để /explain tự tính rule_hits) ----
RULE_PATTERNS = {
    "tautology": re.compile(r"(?i)\b(or|and)\b\s+\d+\s*=\s*\d+"),
    "union_select": re.compile(r"(?i)\bunion\b[\s\S]*\bselect\b"),
    "sql_comment": re.compile(r"(--|/\*|#)"),
    "stacked_danger": re.compile(r";\s*(?:drop|delete|insert|update|alter|create)\b", re.I),
    "time_function": re.compile(r"(?i)\b(sleep|benchmark|pg_sleep|waitfor)\b\s*\("),
    "long_hex": re.compile(r"0x[0-9a-fA-F]{6,}"),
    "information_schema_recon": re.compile(r"(?i)\binformation_schema\b"),
}


def simple_tokenize(text: str) -> List[str]:
    """
    Tokenize query để vẽ heatmap / reasons.
    - lower()
    - normalize dạng "1 = 1" -> "1=1" để ra token đẹp
    """
    text = (text or "").strip().lower()
    if not text:
        return []
    text = re.sub(r"(\d+)\s*=\s*(\d+)", r"\1=\2", text)
    return _TOKEN_RE.findall(text)


def token_weights(tokens: List[str]) -> List[float]:
    """
    Weight trong [0..1]
    - token nghi vấn -> weight cao
    - token thường -> weight thấp

    Gợi ý tinh chỉnh (nếu benign bị đỏ nhiều):
    - Với token SQL phổ thông (select/where/update/insert/...) hạ xuống 0.2~0.35
      thay vì cho 0.85.
    """
    if not tokens:
        return []

    weights: List[float] = []
    for t in tokens:
        tt = (t or "").lower()
        base = 0.10

        # ✅ token nghi vấn
        if tt in SUSPECT_TOKENS:
            base = 0.85

        # ✅ các dấu hiệu mạnh hơn
        if tt in ("--", "/*", "*/", "#"):
            base = max(base, 0.95)

        if tt == ";":
            base = max(base, 0.90)

        # tautology như 1=1
        if tt == "1=1" or re.fullmatch(r"\d+=\d+", tt):
            base = max(base, 0.92)

        # hex blob
        if tt.startswith("0x") and len(tt) >= 8:
            base = max(base, 0.80)

        weights.append(base)

    # normalize để ra [0..1]
    m = max(weights) if weights else 1.0
    if m <= 0:
        return [0.0 for _ in weights]
    return [round(w / m, 3) for w in weights]


def top_tokens(tokens: List[str], weights: List[float], k: int = 5) -> List[Dict]:
    pairs = [{"token": t, "weight": float(w)} for t, w in zip(tokens, weights)]
    pairs.sort(key=lambda x: x["weight"], reverse=True)
    return pairs[:k]


def match_rules(query: str) -> List[str]:
    """Dò tất cả rule hits (có thể nhiều)."""
    q = (query or "").strip()
    if not q:
        return []
    hits: List[str] = []
    for name, rx in RULE_PATTERNS.items():
        if rx.search(q):
            hits.append(name)
    return hits


def pick_primary_rule(rule_hits: List[str]) -> Optional[str]:
    """Chọn 1 rule chính để hiển thị (ưu tiên mức độ nặng)."""
    if not rule_hits:
        return None
    priority = [
        "stacked_danger",
        "union_select",
        "time_function",
        "tautology",
        "sql_comment",
        "long_hex",
        "information_schema_recon",
    ]
    for p in priority:
        if p in rule_hits:
            return p
    return rule_hits[0]


def build_reasons(
    query: str,
    rule_hits: List[str],
    proba: Optional[float],
    threshold: float
) -> List[str]:
    """
    reasons dạng:
    ["RULE:UNION_SELECT", "MODEL:HIGH_PROB", "TOKEN:or", "TOKEN:--", ...]
    """
    reasons: List[str] = []

    # rule reasons (có thể nhiều)
    for r in (rule_hits or []):
        if isinstance(r, str) and r:
            reasons.append(RULE_CODE_MAP.get(r, f"RULE:{r.upper()}"))

    # model reason
    if proba is not None:
        try:
            p = float(proba)
        except Exception:
            p = None
        if p is not None:
            reasons.append("MODEL:HIGH_PROB" if p >= float(threshold) else "MODEL:LOW_PROB")

    # token reasons
    toks = simple_tokenize(query)
    for t in toks:
        if t in SUSPECT_TOKENS:
            reasons.append(f"TOKEN:{t}")

    # unique + giới hạn
    uniq: List[str] = []
    for r in reasons:
        if r not in uniq:
            uniq.append(r)
    return uniq[:12]


def calc_risk_score(proba: Optional[float], rule_hits: List[str]) -> int:
    """
    risk_score 0..100:
    - model đóng góp tối đa 70 điểm
    - rule đóng góp tối đa 30 điểm (nhiều rule thì cộng thêm chút, vẫn cap 30)
    """
    score = 0.0

    if proba is not None:
        try:
            p = float(proba)
        except Exception:
            p = 0.0
        p = min(max(p, 0.0), 1.0)
        score += p * 70.0

    if rule_hits:
        score += min(30.0, 10.0 * len(rule_hits))

    return int(round(min(score, 100.0)))
