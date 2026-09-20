"""
Authentication (blueprint §13, §20). JWT (HS256) + PBKDF2 password hashing, using
only the standard library so there's no extra dependency to install. This is a
prototype auth layer — enough to attach an attributable identity to every action
(which the audit log and human review need), not a production IdP.

Demo users (change before exposing anything):
  analyst / analyst123   → role "analyst"
  admin   / admin123      → role "admin"
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

from evident.config import settings


# --------------------------------------------------------------------------- #
# Password hashing (PBKDF2-HMAC-SHA256)
# --------------------------------------------------------------------------- #

def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or hashlib.sha256(str(time.time()).encode()).digest()[:16]
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 120_000)
    return f"pbkdf2${base64.b64encode(salt).decode()}${base64.b64encode(dk).decode()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, salt_b64, dk_b64 = stored.split("$")
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(dk_b64)
    except Exception:
        return False
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 120_000)
    return hmac.compare_digest(dk, expected)


# --------------------------------------------------------------------------- #
# Minimal JWT (HS256)
# --------------------------------------------------------------------------- #

def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def create_token(sub: str, role: str, ttl_seconds: int = 3600) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {"sub": sub, "role": role, "exp": int(time.time()) + ttl_seconds}
    seg = _b64url(json.dumps(header).encode()) + "." + _b64url(json.dumps(payload).encode())
    sig = hmac.new(settings.jwt_secret.encode(), seg.encode(), hashlib.sha256).digest()
    return seg + "." + _b64url(sig)


def decode_token(token: str) -> dict | None:
    try:
        seg, sig_b64 = token.rsplit(".", 1)
        expected = hmac.new(settings.jwt_secret.encode(), seg.encode(), hashlib.sha256).digest()
        if not hmac.compare_digest(_b64url_decode(sig_b64), expected):
            return None
        payload = json.loads(_b64url_decode(seg.split(".")[1]))
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Demo user table (in-memory)
# --------------------------------------------------------------------------- #

USERS = {
    "analyst": {"role": "analyst", "pw": hash_password("analyst123")},
    "admin": {"role": "admin", "pw": hash_password("admin123")},
}


def authenticate(username: str, password: str) -> dict | None:
    u = USERS.get(username)
    if u and verify_password(password, u["pw"]):
        return {"sub": username, "role": u["role"]}
    return None
