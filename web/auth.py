"""Parola özeti (scrypt), oturum çerezi, çalıştırıcı anahtarları ve çalıştırıcı eşleştirme kodları."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path

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
    return db.one("SELECT u.id, u.username, u.admin, u.server_access FROM sessions s JOIN users u ON u.id = s.user_id "
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


# ------------------------------------------------------------------ sunucu çalıştırıcısı
SERVER_RUNNER_NAME = "Sunucu (CPU)"


def ensure_server_runner(db: DB, token_file: Path) -> int | None:
    """Sunucu çalıştırıcısının kaydını ve anahtarını hazırlar. Anahtar yalnız iki konteynerin paylaştığı dosyaya
    yazılır; dosyadaki anahtar kayıtla uyuşmuyorsa yenilenir. Birden çok sunucu işçisi aynı anda çağırabilir."""
    import fcntl

    owner = db.one("SELECT id FROM users WHERE admin = 1 ORDER BY id LIMIT 1")
    if not owner:
        return None
    token_file.parent.mkdir(parents=True, exist_ok=True)
    with open(token_file.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        r = db.one("SELECT * FROM runners WHERE kind = 'server'")
        try:
            token = token_file.read_text().strip()
        except OSError:
            token = ""
        if r and token and hmac.compare_digest(r["token_hash"], token_hash(token)):
            return r["id"]
        token = "engr_" + secrets.token_urlsafe(32)
        if r:
            db.run("UPDATE runners SET token_hash = ? WHERE id = ?", token_hash(token), r["id"])
            rid = r["id"]
        else:
            rid = db.run("INSERT INTO runners (user_id, name, token_hash, created, kind) VALUES (?, ?, ?, ?, 'server')",
                         owner["id"], SERVER_RUNNER_NAME, token_hash(token), now())
        tmp = token_file.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            fh.write(token)
        os.replace(tmp, token_file)
        return rid


# ------------------------------------------------------------------ eşleştirme (anahtarı elle kopyalamadan bağlanma)
# Çalıştırıcı kısa bir kod ve gizli bir sorgu anahtarı alır; kullanıcı web'de oturum açıkken kodu onaylar; çalıştırıcı
# gizli anahtarla sorar ve çalıştırıcı anahtarını bir kez alır. Kod kısa ömürlü ve tek kullanımlıktır.
PAIR_SECONDS = 600
PAIR_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"     # karışan I, O, 0, 1 yok
PAIR_MAX_PENDING = 200


def normalize_code(code: str) -> str:
    return "".join(ch for ch in (code or "").upper() if ch in PAIR_ALPHABET)


def start_pairing(db: DB, info: dict) -> dict:
    """Yeni eşleştirme -> {code: "ABCD-EFGH", secret, expires_in}. Yalnız özetler saklanır."""
    db.run("DELETE FROM runner_pairings WHERE expires < ?", time.time())
    if db.one("SELECT COUNT(*) AS n FROM runner_pairings")["n"] >= PAIR_MAX_PENDING:
        raise ValueError("çok fazla bekleyen eşleştirme; birkaç dakika sonra tekrar deneyin")
    code = "".join(secrets.choice(PAIR_ALPHABET) for _ in range(8))
    secret = secrets.token_urlsafe(32)
    db.run("INSERT INTO runner_pairings (code_hash, secret_hash, info, expires) VALUES (?, ?, ?, ?)",
           token_hash(code), token_hash(secret), json.dumps(info), time.time() + PAIR_SECONDS)
    return {"code": f"{code[:4]}-{code[4:]}", "secret": secret, "expires_in": PAIR_SECONDS}


def pending_pairing(db: DB, code: str) -> dict | None:
    code = normalize_code(code)
    if len(code) != 8:
        return None
    p = db.one("SELECT * FROM runner_pairings WHERE code_hash = ? AND status = 'pending' AND expires > ?",
               token_hash(code), time.time())
    if p:
        p["info"] = json.loads(p["info"] or "{}")
    return p


def decide_pairing(db: DB, pairing_id: int, user_id: int, name: str, approve: bool, shared: bool = False) -> bool:
    with db.conn() as c:
        return c.execute("UPDATE runner_pairings SET status = ?, user_id = ?, name = ?, shared = ? "
                         "WHERE id = ? AND status = 'pending' AND expires > ?",
                         ("approved" if approve else "denied", user_id, name.strip()[:80], int(shared), pairing_id,
                          time.time())).rowcount == 1


def claim_pairing(db: DB, secret: str) -> dict:
    """Çalıştırıcının sorusu -> {status: pending | approved (+ token) | denied | expired}. Onaylıysa çalıştırıcı
    burada oluşturulur ve anahtar bir kez döner (anahtar veritabanında hiç açık durmaz)."""
    p = db.one("SELECT * FROM runner_pairings WHERE secret_hash = ?", token_hash(secret or ""))
    if not p or p["expires"] < time.time():
        return {"status": "expired"}
    if p["status"] == "pending":
        return {"status": "pending"}
    with db.conn() as c:
        if c.execute("DELETE FROM runner_pairings WHERE id = ?", (p["id"],)).rowcount != 1:
            return {"status": "expired"}
    if p["status"] != "approved":
        return {"status": "denied"}
    rid, token = create_runner(db, p["user_id"], p["name"] or "çalıştırıcı")
    db.run("UPDATE runners SET info = ?, shared = ? WHERE id = ?", p["info"], int(p.get("shared") or 0), rid)
    return {"status": "approved", "token": token, "runner_id": rid, "name": p["name"]}
