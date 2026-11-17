"""Parola özeti (scrypt), oturum çerezi ve çalıştırıcı anahtarları."""
from __future__ import annotations

import hashlib
import hmac
import secrets
import time

from .db import DB, now

SESSION_COOKIE = "eng_session"
SESSION_DAYS = 14


def hash_password(pw: str) -> str:
    salt = secrets.token_bytes(16)
    h = hashlib.scrypt(pw.encode(), salt=salt, n=2 ** 14, r=8, p=1)
    return f"scrypt${salt.hex()}${h.hex()}"


def check_password(pw: str, stored: str) -> bool:
    try:
        _, salt, h = stored.split("$")
    except ValueError:
        return False
    got = hashlib.scrypt(pw.encode(), salt=bytes.fromhex(salt), n=2 ** 14, r=8, p=1)
    return hmac.compare_digest(got.hex(), h)


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create_user(db: DB, username: str, password: str, admin: bool = False) -> int:
    username = username.strip()
    if not username or len(password) < 8:
        raise ValueError("kullanıcı adı boş olamaz, parola en az 8 karakter olmalı")
    if db.one("SELECT id FROM users WHERE username = ?", username):
        raise ValueError(f"{username} adlı kullanıcı zaten var")
    return db.run("INSERT INTO users (username, pw, admin, created) VALUES (?, ?, ?, ?)",
                  username, hash_password(password), int(admin), now())


def login(db: DB, username: str, password: str) -> str | None:
    u = db.one("SELECT * FROM users WHERE username = ?", username.strip())
    if not u or not check_password(password, u["pw"]):
        return None
    token = secrets.token_urlsafe(32)
    db.run("DELETE FROM sessions WHERE expires < ?", time.time())
    db.run("INSERT INTO sessions (token, user_id, expires) VALUES (?, ?, ?)", token_hash(token), u["id"],
           time.time() + SESSION_DAYS * 86400)
    return token


def session_user(db: DB, token: str | None) -> dict | None:
    if not token:
        return None
    return db.one("SELECT u.id, u.username, u.admin FROM sessions s JOIN users u ON u.id = s.user_id "
                  "WHERE s.token = ? AND s.expires > ?", token_hash(token), time.time())


def logout(db: DB, token: str | None) -> None:
    if token:
        db.run("DELETE FROM sessions WHERE token = ?", token_hash(token))


def create_runner(db: DB, user_id: int, name: str) -> tuple[int, str]:
    """Yeni çalıştırıcı -> (numara, anahtar). Anahtar yalnız bir kez gösterilir; veritabanında özeti durur."""
    token = "engr_" + secrets.token_urlsafe(32)
    rid = db.run("INSERT INTO runners (user_id, name, token_hash, created) VALUES (?, ?, ?, ?)",
                 user_id, name.strip() or "çalıştırıcı", token_hash(token), now())
    return rid, token


def runner_by_token(db: DB, token: str | None) -> dict | None:
    return db.one("SELECT * FROM runners WHERE token_hash = ?", token_hash(token)) if token else None
