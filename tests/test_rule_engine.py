"""
Unit test cho rule engine (không cần Docker, không cần model ML).
Import trực tiếp explain_utils.py và reason_codes.py từ services/waf/
(sys.path đã được conftest.py trỏ vào services/waf/).
"""
import pytest

from explain_utils import match_rules, calc_risk_score
from reason_codes import extract_evidence

# Các câu SQLi mà CẢ HAI rule engine (explain_utils.RULE_PATTERNS
# và reason_codes.PATTERNS) đều có regex tương ứng để bắt.
ATTACK_CASES = [
    ("SELECT * FROM users WHERE username='admin' OR 1=1 --", "tautology"),
    ("1' UNION SELECT username, password FROM users--", "union_select"),
    ("'; DROP TABLE users; --", "stacked_danger"),
    ("1' AND SLEEP(5)--", "time_function"),
    ("admin'--", "sql_comment"),
]

# Các câu benign hợp lệ, không được có bất kỳ rule nào bắt nhầm.
BENIGN_CASES = [
    "SELECT * FROM products WHERE id = 5",
    "SELECT name, price FROM products WHERE category='electronics'",
    "UPDATE users SET last_login = NOW() WHERE id = 42",
    "INSERT INTO logs (message) VALUES ('user logged in')",
    "SELECT COUNT(*) FROM orders WHERE status='completed'",
]


@pytest.mark.parametrize("query,expected_rule", ATTACK_CASES)
def test_match_rules_detects_sqli(query, expected_rule):
    hits = match_rules(query)
    assert expected_rule in hits, f"rule '{expected_rule}' không được bắt, hits={hits}, query={query!r}"


@pytest.mark.parametrize("query", BENIGN_CASES)
def test_match_rules_no_false_positive(query):
    hits = match_rules(query)
    assert hits == [], f"false positive: hits={hits} cho query benign {query!r}"


@pytest.mark.parametrize("query,expected_rule", ATTACK_CASES)
def test_extract_evidence_flags_attack(query, expected_rule):
    result = extract_evidence(query)
    assert result.attack_type != "benign", f"reason_codes không flag attack cho {query!r}"
    assert len(result.reasons) > 0


@pytest.mark.parametrize("query", BENIGN_CASES)
def test_extract_evidence_benign(query):
    result = extract_evidence(query)
    assert result.attack_type == "benign", f"reason_codes false positive cho {query!r}: {result.attack_type}"
    assert result.reasons == []


def test_calc_risk_score_attack_higher_than_benign():
    attack_score = calc_risk_score(proba=0.95, rule_hits=["union_select", "sql_comment"])
    benign_score = calc_risk_score(proba=0.01, rule_hits=[])
    assert attack_score > benign_score


# ---------------------------------------------------------------------------
# Hai case dưới đây trước kia lệch giữa hai rule engine (explain_utils thiếu
# "information_schema", reason_codes thiếu "hex blob"). Sau khi bổ sung rule
# tương đương ở cả hai file, cả hai engine phải cùng bắt được cả hai case.
# ---------------------------------------------------------------------------

def test_match_rules_detects_long_hex_blob():
    query = "SELECT * FROM users WHERE id=0x41424344454647"
    hits = match_rules(query)
    assert "long_hex" in hits


def test_extract_evidence_detects_long_hex_blob():
    query = "SELECT * FROM users WHERE id=0x41424344454647"
    result = extract_evidence(query)
    assert result.attack_type != "benign"
    assert any(r.code == "R7" for r in result.reasons)


def test_extract_evidence_detects_information_schema_recon():
    query = "SELECT table_name FROM information_schema.tables"
    result = extract_evidence(query)
    assert result.attack_type != "benign"
    assert any(r.code == "R6" for r in result.reasons)


def test_match_rules_detects_information_schema_recon():
    query = "SELECT table_name FROM information_schema.tables"
    hits = match_rules(query)
    assert "information_schema_recon" in hits
