"""Web uygulamasının ayarları (ortam değişkenleri; Docker'da deploy/.env).

    ENG_DATA_DIR               görüntüler, veritabanı (varsayılan: <proje>/webdata)
    LABEL_STUDIO_URL           sunucunun Label Studio'ya eriştiği adres (Docker'da http://label-studio:8080)
    LABEL_STUDIO_PUBLIC_URL    tarayıcıdaki bağlantılar için Label Studio adresi (varsayılan: LABEL_STUDIO_URL)
    LABEL_STUDIO_API_KEY       Label Studio anahtarı (hizmet hesabı; legacy token ya da Personal Access Token)
    ENG_ADMIN_USERNAME / ENG_ADMIN_PASSWORD   veritabanında hiç kullanıcı yoksa ilk yönetici bunlarla oluşturulur
    ENG_PUBLIC_URL             web uygulamasının dış adresi (çalıştırıcı paketine ve eşleştirme bağlantısına yazılır)
    ENG_SECURE_COOKIES         1 ise oturum çerezi yalnız HTTPS'te gönderilir (Caddy arkasında 1)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from app import config as C


def _bool(v: str | None) -> bool:
    return str(v or "").strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("ENG_DATA_DIR", C.ROOT / "webdata")))
    ls_url: str = field(default_factory=lambda: os.environ.get("LABEL_STUDIO_URL", "http://localhost:8080"))
    ls_public_url: str = field(default_factory=lambda: os.environ.get("LABEL_STUDIO_PUBLIC_URL", ""))
    ls_api_key: str = field(default_factory=lambda: os.environ.get("LABEL_STUDIO_API_KEY", ""))
    admin_username: str = field(default_factory=lambda: os.environ.get("ENG_ADMIN_USERNAME", ""))
    admin_password: str = field(default_factory=lambda: os.environ.get("ENG_ADMIN_PASSWORD", ""))
    public_url: str = field(default_factory=lambda: os.environ.get("ENG_PUBLIC_URL", ""))
    secure_cookies: bool = field(default_factory=lambda: _bool(os.environ.get("ENG_SECURE_COOKIES")))
    classes: dict = field(default_factory=lambda: dict(C.DEFAULTS["classes"]))
    label_config: str = field(default_factory=lambda: C.LABEL_CONFIG.read_text(encoding="utf-8"))

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.ls_url = self.ls_url.rstrip("/")
        self.ls_public_url = (self.ls_public_url or self.ls_url).rstrip("/")

    @property
    def db_path(self) -> Path:
        return self.data_dir / "eng.sqlite3"

    def dataset_dir(self, dataset_id: int) -> Path:
        return self.data_dir / "datasets" / str(dataset_id)
