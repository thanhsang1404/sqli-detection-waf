# reason_codes.py
import re
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Reason:
    code: str
    title: str
    detail: str
    severity: int  # 1..5

@dataclass
class EvidenceResult:
    attack_type: str
    reasons: List[Reason]

# Một số regex cơ bản (bạn có thể mở rộng)
PATTERNS = [
    ("R1", "UNION-based pattern", r"\bunion\b\s+\bselect\b", 5, "union-based"),
    ("R2", "SQL comment marker", r"(--|#|/\*|\*/)", 4, "comment/obfuscation"),
    ("R3", "Tautology OR 1=1", r"\bor\b\s+1\s*=\s*1\b", 5, "tautology"),
    ("R4", "Stacked query delimiter", r";\s*(drop|alter|truncate|insert|update|delete)\b", 5, "stacked"),
    ("R5", "Time-based keyword", r"\b(sleep|benchmark|pg_sleep|waitfor\s+delay)\b", 5, "time-based"),
    ("R6", "Information schema access", r"\binformation_schema\b", 3, "recon"),
    ("R7", "Hex-encoded blob", r"0x[0-9a-fA-F]{6,}", 3, "hex-encoded"),
]

def normalize_query(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())

def extract_evidence(query: str, top_k: int = 3) -> EvidenceResult:
    q = normalize_query(query)
    reasons: List[Reason] = []
    type_votes = {}

    for code, title, pattern, sev, atype in PATTERNS:
        if re.search(pattern, q, flags=re.IGNORECASE):
            reasons.append(Reason(
                code=code,
                title=title,
                detail=f"Matched pattern: {pattern}",
                severity=sev,
            ))
            type_votes[atype] = type_votes.get(atype, 0) + sev

    # Attack type dựa trên vote severity
    attack_type = "benign"
    if type_votes:
        attack_type = sorted(type_votes.items(), key=lambda x: x[1], reverse=True)[0][0]

    # Lấy top_k theo severity
    reasons.sort(key=lambda r: r.severity, reverse=True)
    return EvidenceResult(attack_type=attack_type, reasons=reasons[:top_k])

def risk_level_from(prob: float) -> str:
    # risk theo xác suất model (bạn có thể chỉnh)
    if prob >= 0.95:
        return "Critical"
    if prob >= 0.90:
        return "High"
    if prob >= 0.50:
        return "Medium"
    return "Low"
