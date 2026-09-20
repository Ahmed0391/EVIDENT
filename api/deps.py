"""RBAC dependencies (blueprint §13). Analyst vs Admin, as FastAPI dependencies."""
from __future__ import annotations

from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer

from evident.api.auth import decode_token

oauth2 = OAuth2PasswordBearer(tokenUrl="auth/login", auto_error=False)


def current_user(token: str | None = Depends(oauth2)) -> dict:
    payload = decode_token(token or "")
    if payload is None:
        raise HTTPException(401, "not authenticated (get a token from POST /auth/login)")
    return payload


def require_role(*roles: str):
    def _dep(user: dict = Depends(current_user)) -> dict:
        if user["role"] not in roles:
            raise HTTPException(403, f"requires role in {roles}; you are '{user['role']}'")
        return user
    return _dep
