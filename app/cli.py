"""TUI'siz komut satırı sürümü.

    python -m app.cli check                 # Label Studio bağlantısı ve model paketi
    python -m app.cli predict [--images DIR]
    python -m app.cli upload
    python -m app.cli status
    python -m app.cli export [--format masks coco yolo]
    python -m app.cli evaluate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config as C, core, export
from .ls_client import LSError
from .model import ModelError, from_config


def _progress(n: int, total: int, name: str) -> None:
    print(f"[{n}/{total}] {name}", file=sys.stderr)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="eng-label-app")
    ap.add_argument("--config", default=str(C.CONFIG))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    p = sub.add_parser("predict")
    p.add_argument("--images", help="görüntü klasörü (varsayılan: config data.images)")
    sub.add_parser("upload")
    sub.add_parser("status")
    e = sub.add_parser("export")
    e.add_argument("--format", nargs="+", choices=export.FORMATS, default=list(export.FORMATS))
    sub.add_parser("evaluate")
    a = ap.parse_args(argv)
    cfg = C.load(Path(a.config))
    try:
        if a.cmd == "check":
            m = from_config(cfg)
            print(f"model: {m.version} ({m.bundle})")
            print(f"Label Studio: {cfg['label_studio']['url']} -> {core.client(cfg).check()}")
        elif a.cmd == "predict":
            imgs = C.list_images(a.images or cfg["data"]["images"])
            rows = core.run_predict(cfg, imgs, _progress, print)
            print(json.dumps(rows, indent=1))
        elif a.cmd == "upload":
            core.run_upload(cfg, _progress, print)
        elif a.cmd == "status":
            s = core.run_status(cfg)
            print(f"proje {s['project']}: toplam {s['total']}, kontrol edilen {s['labeled']}, bekleyen {s['pending']}")
            print(s["link"])
        elif a.cmd == "export":
            core.run_export(cfg, a.format, _progress, print)
        elif a.cmd == "evaluate":
            summary, _, out = core.run_evaluate(cfg, _progress, print)
            print((out / "report.md").read_text())
    except (LSError, ModelError, FileNotFoundError) as e:
        print(f"hata: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
