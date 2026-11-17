"""SQLite deposu: kullanıcılar, oturumlar, çalıştırıcılar (ve eşleştirme kodları), veri setleri, görüntüler, işler ve raporlar.

Etiketlerin kendisi Label Studio'da durur; burada yalnız hangi görüntünün hangi göreve karşılık geldiği, işlerin durumu
ve kaydedilmiş karşılaştırma raporları tutulur. Her istek kendi bağlantısını açar (WAL kipi).
"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL, pw TEXT NOT NULL, admin INTEGER NOT NULL DEFAULT 0,
    created TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sessions (
    token TEXT PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE, expires REAL NOT NULL);
CREATE TABLE IF NOT EXISTS runners (
    id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE, name TEXT NOT NULL,
    token_hash TEXT UNIQUE NOT NULL, models TEXT NOT NULL DEFAULT '[]', info TEXT NOT NULL DEFAULT '{}',
    last_seen REAL, created TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS runner_pairings (
    id INTEGER PRIMARY KEY, code_hash TEXT UNIQUE NOT NULL, secret_hash TEXT UNIQUE NOT NULL,
    info TEXT NOT NULL DEFAULT '{}', status TEXT NOT NULL DEFAULT 'pending',
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE, name TEXT, expires REAL NOT NULL);
CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name TEXT NOT NULL, version TEXT NOT NULL, card TEXT NOT NULL, weights TEXT NOT NULL,
    created TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY, name TEXT NOT NULL, owner_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    ls_project INTEGER, created TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY, dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE, name TEXT NOT NULL,
    file TEXT NOT NULL, display TEXT, width INTEGER, height INTEGER, ls_task INTEGER, created TEXT NOT NULL,
    UNIQUE (dataset_id, name));
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY, dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL, runner_id INTEGER REFERENCES runners(id) ON DELETE SET NULL,
    model TEXT NOT NULL, model_version TEXT NOT NULL, remote_model INTEGER REFERENCES models(id) ON DELETE SET NULL,
    min_score REAL NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'queued', error TEXT, created TEXT NOT NULL, started TEXT, finished TEXT);
CREATE TABLE IF NOT EXISTS job_items (
    job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'pending', summary TEXT, error TEXT, seconds REAL, PRIMARY KEY (job_id, image_id));
CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY, dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL, model_version TEXT, summary TEXT NOT NULL,
    rows TEXT NOT NULL, created TEXT NOT NULL);
"""

JOB_ACTIVE = ("queued", "running")


def now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


class DB:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.conn() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.executescript(SCHEMA)
            if "remote_model" not in [r[1] for r in c.execute("PRAGMA table_info(jobs)")]:
                c.execute("ALTER TABLE jobs ADD COLUMN remote_model INTEGER REFERENCES models(id) ON DELETE SET NULL")
            # paylaşılmayan çalıştırıcıya yalnız sahibi iş verebilir (eski kayıtlar özel kalır)
            if "shared" not in [r[1] for r in c.execute("PRAGMA table_info(runners)")]:
                c.execute("ALTER TABLE runners ADD COLUMN shared INTEGER NOT NULL DEFAULT 0")
            if "shared" not in [r[1] for r in c.execute("PRAGMA table_info(runner_pairings)")]:
                c.execute("ALTER TABLE runner_pairings ADD COLUMN shared INTEGER NOT NULL DEFAULT 0")
            # sunucu çalıştırıcısı (kind = 'server') ve kullanıcının ona erişim izni ('' | requested | approved | denied)
            if "kind" not in [r[1] for r in c.execute("PRAGMA table_info(runners)")]:
                c.execute("ALTER TABLE runners ADD COLUMN kind TEXT NOT NULL DEFAULT 'local'")
            ucols = [r[1] for r in c.execute("PRAGMA table_info(users)")]
            if "server_access" not in ucols:
                c.execute("ALTER TABLE users ADD COLUMN server_access TEXT NOT NULL DEFAULT ''")
                c.execute("ALTER TABLE users ADD COLUMN server_note TEXT")

    @contextmanager
    def conn(self):
        c = sqlite3.connect(self.path, timeout=30)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys=ON")
        try:
            yield c
            c.commit()
        finally:
            c.close()

    def one(self, sql: str, *args) -> dict | None:
        with self.conn() as c:
            r = c.execute(sql, args).fetchone()
        return dict(r) if r else None

    def all(self, sql: str, *args) -> list[dict]:
        with self.conn() as c:
            return [dict(r) for r in c.execute(sql, args).fetchall()]

    def run(self, sql: str, *args) -> int:
        """INSERT/UPDATE/DELETE -> son eklenen satır numarası."""
        with self.conn() as c:
            return c.execute(sql, args).lastrowid

    # -------------------------------------------------------------- sık kullanılan sorgular
    def dataset(self, dataset_id: int) -> dict | None:
        return self.one("SELECT d.*, u.username AS owner FROM datasets d LEFT JOIN users u ON u.id = d.owner_id "
                        "WHERE d.id = ?", dataset_id)

    def images(self, dataset_id: int) -> list[dict]:
        return self.all("SELECT * FROM images WHERE dataset_id = ? ORDER BY name", dataset_id)

    def runner_models(self, runner: dict) -> list[dict]:
        return json.loads(runner["models"] or "[]")

    def jobs(self, dataset_id: int | None = None, limit: int = 50) -> list[dict]:
        where, args = ("WHERE j.dataset_id = ?", (dataset_id,)) if dataset_id else ("", ())
        return self.all(
            "SELECT j.*, d.name AS dataset, r.name AS runner, u.username AS user, "
            "(SELECT COUNT(*) FROM job_items i WHERE i.job_id = j.id) AS total, "
            "(SELECT COUNT(*) FROM job_items i WHERE i.job_id = j.id AND i.status = 'done') AS done, "
            "(SELECT COUNT(*) FROM job_items i WHERE i.job_id = j.id AND i.status = 'error') AS failed "
            f"FROM jobs j JOIN datasets d ON d.id = j.dataset_id LEFT JOIN runners r ON r.id = j.runner_id "
            f"LEFT JOIN users u ON u.id = j.user_id {where} ORDER BY j.id DESC LIMIT {int(limit)}", *args)

    def predicted_versions(self, image_id: int) -> list[str]:
        """Bu görüntü için ön etiketi başarıyla yüklenmiş model sürümleri."""
        return [r["model_version"] for r in self.all(
            "SELECT DISTINCT j.model_version FROM job_items i JOIN jobs j ON j.id = i.job_id "
            "WHERE i.image_id = ? AND i.status = 'done'", image_id)]
