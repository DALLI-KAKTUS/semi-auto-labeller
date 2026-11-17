"""config.yaml okuma/yazma ve çalışma klasörü yolları."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config.yaml"
LABEL_CONFIG = ROOT / "labeling_config.xml"
IMAGE_EXT = (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".czi")

DEFAULTS = {
    "label_studio": {"url": "http://localhost:8080", "api_key": "", "project_title": "PC12 hücre etiketleme",
                     "project_id": None},
    "model": {"card": "models/lite_v8.json", "weights": "", "device": "auto"},
    "classes": {"cell": "Cell", "branch": "Neurite", "group": "Group"},
    "min_score": 0.0,
    "data": {"images": ""},
    "work_dir": "work",
}


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _merge(base[k], v) if isinstance(base.get(k), dict) and isinstance(v, dict) else v
    return out


def load(path: Path = CONFIG) -> dict:
    cfg = _merge(DEFAULTS, yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {})
    cfg["_path"] = str(path)
    cfg["_key_from_env"] = not cfg["label_studio"]["api_key"]
    if cfg["_key_from_env"]:
        cfg["label_studio"]["api_key"] = os.environ.get("LABEL_STUDIO_API_KEY", "")
    return cfg


def save(cfg: dict) -> None:
    """Yüklendiği dosyaya yazar; ortam değişkeninden gelen API anahtarı dosyaya yazılmaz."""
    data = {k: v for k, v in cfg.items() if not k.startswith("_")}
    if cfg.get("_key_from_env"):
        data["label_studio"] = dict(data["label_studio"], api_key="")
    Path(cfg.get("_path", CONFIG)).write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
                                              encoding="utf-8")


def work_dir(cfg: dict) -> Path:
    p = Path(cfg["work_dir"])
    return p if p.is_absolute() else ROOT / p


def pred_dir(cfg: dict) -> Path:
    """Model çıktıları: <stem>.json (Label Studio prediction) + <stem>.npz (maskeler)."""
    return work_dir(cfg) / "predictions"


def upload_dir(cfg: dict) -> Path:
    """Label Studio'nun gösteremediği girdiler (CZI) için PNG kopyaları."""
    return work_dir(cfg) / "upload"


def list_images(folder: str | Path) -> list[Path]:
    folder = Path(folder).expanduser()
    if not folder.is_dir():
        raise FileNotFoundError(f"görüntü klasörü yok: {folder}")
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXT and not p.name.startswith("."))
