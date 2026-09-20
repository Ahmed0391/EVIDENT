import time

from evident.api import auth


def test_password_hash_roundtrip():
    h = auth.hash_password("secret123")
    assert auth.verify_password("secret123", h) is True
    assert auth.verify_password("wrong", h) is False


def test_jwt_roundtrip():
    tok = auth.create_token("analyst", "analyst")
    payload = auth.decode_token(tok)
    assert payload and payload["sub"] == "analyst" and payload["role"] == "analyst"


def test_jwt_rejects_tampering():
    tok = auth.create_token("analyst", "analyst")
    assert auth.decode_token(tok + "x") is None
    assert auth.decode_token("not.a.token") is None


def test_jwt_rejects_expired():
    tok = auth.create_token("analyst", "analyst", ttl_seconds=-1)
    time.sleep(0.01)
    assert auth.decode_token(tok) is None


def test_authenticate_demo_users():
    assert auth.authenticate("analyst", "analyst123")["role"] == "analyst"
    assert auth.authenticate("admin", "admin123")["role"] == "admin"
    assert auth.authenticate("analyst", "nope") is None
