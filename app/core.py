"""Akışın adımları (TUI ve komut satırı ortak kullanır). Her adım ilerleme ve kayıt için geri çağırma alır.

    tahmin  -> work/predictions/<stem>.json (+ .npz maskeler)
    yükleme -> Label Studio projesi: görüntü + tahmin (ön etiket)
    durum   -> toplam / kontrol edilen / bekleyen görev
    dışa aktarma -> work/export/<zaman>/ (ls_tasks.json, masks/, coco.json, yolo/)
    karşılaştırma -> work/reports/<zaman>/report.json + report.md
"""
from __future__ import annotations

import datetime as dt
import json
import time
import traceback
from pathlib import Path
from typing import Callable

from . import config as C, convert, evaluate, export
from .ls_client import LSClient, LSError, latest_annotation, original_name, prediction_for
from .model import ModelError, from_config, save_masks

Progress = Callable[[int, int, str], None]
Log = Callable[[str], None]


def _noop(*_a, **_k):
    pass


def _stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def client(cfg: dict) -> LSClient:
    ls = cfg["label_studio"]
    return LSClient(ls["url"], ls["api_key"])


def predictions(cfg: dict) -> dict[str, dict]:
    """Kaydedilmiş tahminler: yüklenecek dosya adı -> tahmin kaydı."""
    out = {}
    for p in sorted(C.pred_dir(cfg).glob("*.json")):
        rec = json.loads(p.read_text())
        out[Path(rec["display"]).name] = rec
    return out


# ------------------------------------------------------------------ 1-2. tahmin
def run_predict(cfg: dict, images: list[Path], progress: Progress = _noop, log: Log = _noop) -> list[dict]:
    model = from_config(cfg, C.upload_dir(cfg))
    out_dir = C.pred_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"model: {model.version} ({model.bundle}), cihaz: {model.device}")
    progress(0, len(images), "model yükleniyor")
    model.load()
    rows = []
    for n, path in enumerate(images):
        progress(n, len(images), path.name)
        t0 = time.time()
        try:
            pred = model.predict(path)
            ls = convert.to_ls(pred, cfg["classes"], float(cfg.get("min_score", 0.0)))
        except ModelError:
            raise
        except Exception as e:                       # bir görüntünün hatası tüm işi durdurmasın
            log(f"HATA {path.name}: {type(e).__name__}: {e}")
            (out_dir / f"{path.stem}.error.txt").write_text(traceback.format_exc())
            rows.append({"image": path.name, "error": f"{type(e).__name__}: {e}"})
            continue
        rec = {"source": str(path), "display": str(pred.display), "meta": pred.meta, "created": _stamp(),
               "summary": ls.pop("summary"), "prediction": ls}
        (out_dir / f"{path.stem}.json").write_text(json.dumps(rec))
        save_masks(pred, out_dir / f"{path.stem}.npz")
        row = {"image": path.name, **rec["summary"], "score": ls["score"], "seconds": round(time.time() - t0, 1)}
        rows.append(row)
        log(f"{path.name}: {row['cells']} hücre, {row['groups']} grup, {row['neurite_px']} px nörit, "
            f"skor {row['score']:.3f}, {row['seconds']} s")
    progress(len(images), len(images), "bitti")
    return rows


# ------------------------------------------------------------------ 3-4. Label Studio'ya yükleme
def run_upload(cfg: dict, progress: Progress = _noop, log: Log = _noop, save_config: bool = True) -> int:
    recs = predictions(cfg)
    if not recs:
        raise LSError("yüklenecek tahmin yok: önce tahmin adımını çalıştırın")
    ls = client(cfg)
    log(f"Label Studio: {ls.check()} olarak bağlanıldı")
    pid = ls.get_or_create_project(cfg["label_studio"].get("project_id"), cfg["label_studio"]["project_title"],
                                   C.LABEL_CONFIG.read_text(encoding="utf-8"))
    if save_config and cfg["label_studio"].get("project_id") != pid:
        cfg["label_studio"]["project_id"] = pid
        C.save(cfg)
    existing = ls.tasks_by_name(pid)
    versions = set()
    for n, (name, rec) in enumerate(recs.items()):
        progress(n, len(recs), name)
        pred = rec["prediction"]
        versions.add(pred["model_version"])
        task = existing.get(name)
        if task is None:
            tid = ls.upload_image(pid, Path(rec["display"]))
            log(f"{name}: yüklendi (görev {tid})")
        else:
            tid = task.id
            if prediction_for(task, pred["model_version"]):
                log(f"{name}: görev {tid} bu sürümün tahminine zaten sahip, atlandı")
                continue
        ls.add_prediction(tid, pred)
        log(f"{name}: ön etiket eklendi ({len(pred['result'])} bölge)")
    for v in versions:
        ls.set_model_version(pid, v)
    progress(len(recs), len(recs), "bitti")
    log(f"proje {pid}: {ls.project_link(pid)}")
    return pid


def _project_id(cfg: dict) -> int:
    pid = cfg["label_studio"].get("project_id")
    if not pid:
        raise LSError("proje seçilmedi: önce yükleme yapın ya da Ayarlar'da proje numarasını girin")
    return int(pid)


# ------------------------------------------------------------------ 5. kontrol durumu
def run_status(cfg: dict) -> dict:
    ls, pid = client(cfg), _project_id(cfg)
    tasks = ls.tasks(pid)
    rows = []
    for t in tasks:
        ann, pred = latest_annotation(t), prediction_for(t)
        rows.append({"task": t.id, "image": original_name(t.data), "labeled": ann is not None,
                     "regions_model": len(pred["result"]) if pred else 0,
                     "regions_human": len(ann["result"]) if ann else None,
                     "score": pred["score"] if pred else None})
    labeled = sum(r["labeled"] for r in rows)
    return {"project": pid, "link": ls.project_link(pid), "total": len(rows), "labeled": labeled,
            "pending": len(rows) - labeled, "tasks": rows}


# ------------------------------------------------------------------ 6. dışa aktarma
def task_shape(task, ann: dict, fallback: tuple[int, int] | None = None) -> tuple[int, int]:
    """Görevin görüntü boyutu (h, w): etiket ya da tahmin sonucundan, yoksa verilen yedek değerden."""
    for res in (ann["result"], *(p["result"] for p in [prediction_for(task)] if p)):
        s = convert.result_size(res)
        if s:
            return s
    if fallback:
        return fallback
    raise ValueError(f"görev {task.id}: görüntü boyutu bulunamadı")


def _shape(task, ann: dict, recs: dict) -> tuple[int, int]:
    rec = recs.get(original_name(task.data))
    return task_shape(task, ann, (rec["summary"]["height"], rec["summary"]["width"]) if rec else None)


def compare_task(task, ann: dict, shape: tuple[int, int], classes: dict, model_version: str | None = None) -> dict | None:
    """Bir görevin model tahmini (verilen sürümün ya da en sonuncusu) ile son insan etiketi; tahmin yoksa None."""
    pred = prediction_for(task, model_version)
    if pred is None:
        return None
    r = evaluate.compare(convert.to_masks(pred["result"], shape, classes), convert.to_masks(ann["result"], shape, classes))
    c = r["cells"]
    r.update(task=task.id, image=original_name(task.data), model_version=pred["model_version"],
             unchanged=not (c["deleted"] or c["added"] or c["edited"]) and not any(
                 r[k]["added_px"] or r[k]["removed_px"] for k in ("neurite", "group")))
    return r


def labeled_tasks(cfg: dict) -> list[tuple]:
    """(görev, son insan etiketi) çiftleri; yalnız kontrol edilmiş görevler."""
    ls, pid = client(cfg), _project_id(cfg)
    return [(t, a) for t in ls.tasks(pid) if (a := latest_annotation(t)) is not None]


def run_export(cfg: dict, formats=export.FORMATS, progress: Progress = _noop, log: Log = _noop) -> Path:
    pairs = labeled_tasks(cfg)
    if not pairs:
        raise LSError("dışa aktarılacak kontrol edilmiş görev yok")
    recs = predictions(cfg)
    out = C.work_dir(cfg) / "export" / _stamp()
    out.mkdir(parents=True)
    raw, coco = [], export.Coco(cfg["classes"])
    for n, (task, ann) in enumerate(pairs):
        name = original_name(task.data)
        progress(n, len(pairs), name)
        masks = convert.to_masks(ann["result"], _shape(task, ann, recs), cfg["classes"])
        stem = Path(name).stem
        raw.append({"task": task.id, "image": name, "annotation": ann})
        if "masks" in formats:
            export.write_masks(out / "masks", stem, masks)
        if "coco" in formats:
            coco.add(name, masks)
        if "yolo" in formats:
            rec = recs.get(name)
            export.write_yolo(out / "yolo", stem, Path(rec["display"]) if rec else None, masks, cfg["classes"])
        log(f"{name}: {int(masks['cell_labels'].max())} hücre, {int(masks['branch'].sum())} px nörit")
    if "coco" in formats:
        coco.write(out / "coco.json")
    (out / "ls_tasks.json").write_text(json.dumps(raw, indent=1, default=str))
    progress(len(pairs), len(pairs), "bitti")
    log(f"dışa aktarıldı: {out}")
    return out


# ------------------------------------------------------------------ 7. karşılaştırma
def run_evaluate(cfg: dict, progress: Progress = _noop, log: Log = _noop) -> tuple[dict, list[dict], Path]:
    pairs = labeled_tasks(cfg)
    recs = predictions(cfg)
    rows = []
    for n, (task, ann) in enumerate(pairs):
        name = original_name(task.data)
        progress(n, len(pairs), name)
        r = compare_task(task, ann, _shape(task, ann, recs), cfg["classes"])
        if r is None:
            log(f"{name}: model tahmini yok, atlandı")
            continue
        rows.append(r)
    if not rows:
        raise LSError("karşılaştırılacak görev yok: hem tahmini hem insan etiketi olan görev gerekli")
    summary = evaluate.summarize(rows)
    out = C.work_dir(cfg) / "reports" / _stamp()
    out.mkdir(parents=True)
    (out / "report.json").write_text(json.dumps({"summary": summary, "tasks": rows}, indent=1))
    (out / "report.md").write_text(evaluate.to_markdown(summary, rows), encoding="utf-8")
    progress(len(pairs), len(pairs), "bitti")
    log(f"rapor: {out / 'report.md'}")
    return summary, rows, out
