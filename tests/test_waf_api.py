"""
Test API của WAF (services/waf/waf_tfserving.py) bằng FastAPI TestClient.
Không cần Docker, không gọi network thật ra mock_tfserving:
- Case benign đi qua nhánh is_obviously_benign() -> không gọi model.
- Case SQLi trong test_explain_structure mock hẳn call_model_proba() để
  không phụ thuộc TF-Serving thật.
"""
from fastapi.testclient import TestClient

import waf_tfserving

client = TestClient(waf_tfserving.app)


def test_root_returns_threshold():
    resp = client.get("/")
    assert resp.status_code == 200

    data = resp.json()
    assert "threshold" in data
    assert data["threshold"] == waf_tfserving.THRESHOLD


def test_predict_missing_field_returns_422():
    resp = client.post("/predict", json={})
    assert resp.status_code == 422
    assert "detail" in resp.json()


def test_predict_obviously_benign_allows_without_model_call(monkeypatch):
    def fail_if_called(query_text):
        raise AssertionError("call_model_proba không được gọi cho query obviously-benign")

    monkeypatch.setattr(waf_tfserving, "call_model_proba", fail_if_called)

    resp = client.post("/predict", json={"query": "SELECT name FROM products WHERE id = 1"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["prediction"] == "Benign"


def test_explain_structure(monkeypatch):
    monkeypatch.setattr(waf_tfserving, "call_model_proba", lambda q: (0.97, None))

    payload = {"query": "1' UNION SELECT username, password FROM users--"}
    resp = client.post("/explain", json=payload)
    assert resp.status_code == 200

    data = resp.json()
    for field in ("tokens", "weights", "risk_score", "reasons"):
        assert field in data, f"thiếu field '{field}' trong response /explain"

    assert isinstance(data["tokens"], list)
    assert isinstance(data["weights"], list)
    assert len(data["tokens"]) == len(data["weights"])
    assert isinstance(data["risk_score"], int)
    assert isinstance(data["reasons"], list)
    assert data["prediction"] == "SQLi Attack"
    assert "union_select" in data["rule_hits"]
