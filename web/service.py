"""Web uygulamasının iş mantığı: veri setleri (= Label Studio projeleri), görüntüler (= görevler), çalıştırıcıdan gelen
tahminler, model-insan karşılaştırması, bindirme görüntüsü ve dışa aktarma. Sayfa ve API katmanı (server.py) yalnız
bunu çağırır.
"""
from __future__ import annotations

import io
import hashlib
import json
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app import convert, core, evaluate, export, imageio
from app.ls_client import LSClient, LSError, latest_annotation, prediction_for

from .db import DB, now
from .settings import Settings

THUMB = 240
OVERLAY_MAX = 1600


class UserError(ValueError):
    """Kullanıcıya olduğu gibi gösterilecek hata."""


def safe_name(name: str) -> str:
    name = Path(name.replace("\\", "/")).name.strip()
    name = re.sub(r"[^\w.\- ()]+", "_", name)
    return name[:150] or "image"


class Service:
    def __init__(self, settings: Settings, db: DB, ls_factory=None):
        self.s = settings
        self.db = db
        self._ls_factory = ls_factory or (lambda: LSClient(settings.ls_url, settings.ls_api_key))

    def ls(self) -> LSClient:
        return self._ls_factory()

    def project_link(self, pid: int | None) -> str | None:
        return f"{self.s.ls_public_url}/projects/{pid}/data" if pid else None

    def task_link(self, pid: int | None, tid: int | None) -> str | None:
        return f"{self.s.ls_public_url}/projects/{pid}/data?task={tid}" if pid and tid else None

    def add_model(self, user_id: int, card_file, weights_file) -> dict:
        """Model kartı + ağırlık dosyası depolanır; asla sunucuda çalıştırılmaz. Çalıştırıcı iş sırasında indirir."""
        if not weights_file.filename.lower().endswith((".onnx", ".pt", ".pth")):
            raise UserError("model dosyası .onnx, .pt veya .pth olmalı")
        try:
            card = json.load(card_file.file)
            if not isinstance(card, dict) or not isinstance(card.get("name"), str):
                raise ValueError("name alanı gerekli")
            if card.get("format") not in ("onnx", "smp", "torchscript"):
                raise ValueError("format onnx, smp veya torchscript olmalı")
            if card["format"] == "smp" and not isinstance(card.get("net"), dict):
                raise ValueError("smp modelinde net alanı gerekli")
        except (ValueError, UnicodeDecodeError, AttributeError, TypeError) as e:
            raise UserError(f"model kartı geçersiz: {e}") from e
        name = safe_name(card["name"]).strip(". ")
        if not name:
            raise UserError("model adı geçersiz")
        mid = self.db.run("INSERT INTO models (user_id, name, version, card, weights, created) "
                          "VALUES (?, ?, '', '', '', ?)", user_id, name, now())
        directory = self.s.data_dir / "models" / str(mid)
        directory.mkdir(parents=True)
        weights = directory / ("weights" + Path(weights_file.filename).suffix.lower())
        try:
            digest = hashlib.sha256()
            with weights.open("wb") as out:
                while chunk := weights_file.file.read(1 << 20):
                    out.write(chunk)
                    digest.update(chunk)
            if weights.stat().st_size == 0:
                raise UserError("model dosyası boş")
            card["name"], card["weights"] = name, weights.name
            card_path = directory / "model.json"
            card_path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
            recipe = json.dumps({k: card.get(k) for k in ("format", "net", "input", "output", "decode")}, sort_keys=True)
            version = f"{name}-{digest.hexdigest()[:8]}-{hashlib.sha256(recipe.encode()).hexdigest()[:6]}"
            self.db.run("UPDATE models SET version = ?, card = ?, weights = ? WHERE id = ?", version,
                        str(card_path.relative_to(self.s.data_dir)), str(weights.relative_to(self.s.data_dir)), mid)
        except Exception:
            shutil.rmtree(directory, ignore_errors=True)
            self.db.run("DELETE FROM models WHERE id = ?", mid)
            raise
        return self.db.one("SELECT * FROM models WHERE id = ?", mid)

    # -------------------------------------------------------------- veri setleri
    def create_dataset(self, name: str, user_id: int) -> int:
        name = name.strip()
        if not name:
            raise UserError("veri seti adı boş olamaz")
        pid = self.ls().get_or_create_project(None, name, self.s.label_config)
        return self.db.run("INSERT INTO datasets (name, owner_id, ls_project, created) VALUES (?, ?, ?, ?)",
                           name, user_id, pid, now())

    def delete_dataset(self, ds: dict, delete_project: bool) -> None:
        if delete_project and ds["ls_project"]:
            try:
                self.ls().delete_project(ds["ls_project"])
            except LSError as e:
                if "404" not in str(e):
                    raise
        shutil.rmtree(self.s.dataset_dir(ds["id"]), ignore_errors=True)
        self.db.run("DELETE FROM datasets WHERE id = ?", ds["id"])

    # -------------------------------------------------------------- görüntüler
    def image_dir(self, img: dict) -> Path:
        return self.s.dataset_dir(img["dataset_id"]) / "images" / str(img["id"])

    def add_image(self, ds: dict, filename: str, fileobj) -> dict:
        name = safe_name(filename)
        if Path(name).suffix.lower() not in imageio.EXT:
            raise UserError(f"{name}: desteklenmeyen dosya türü ({', '.join(imageio.EXT)})")
        if self.db.one("SELECT id FROM images WHERE dataset_id = ? AND name = ?", ds["id"], name):
            raise UserError(f"{name}: bu adda bir görüntü zaten var")
        iid = self.db.run("INSERT INTO images (dataset_id, name, file, created) VALUES (?, ?, '', ?)",
                          ds["id"], name, now())
        img = {"id": iid, "dataset_id": ds["id"], "name": name}
        d = self.image_dir(img)
        d.mkdir(parents=True, exist_ok=True)
        path = d / name
        try:
            with open(path, "wb") as fh:
                shutil.copyfileobj(fileobj, fh, 1 << 20)
            if path.suffix.lower() in imageio.BROWSER_EXT:
                display = path
                with Image.open(path) as im:
                    im.verify()
            else:                                           # TIFF / CZI: tarayıcı gösteremez, PNG kopyası
                display = imageio.save_display(path, d / "display" / f"{path.stem}.png")
            with Image.open(display) as im:
                w, h = im.size
                im.thumbnail((THUMB, THUMB))
                im.convert("RGB").save(d / "thumb.jpg", quality=85)
        except Exception as e:
            shutil.rmtree(d, ignore_errors=True)
            self.db.run("DELETE FROM images WHERE id = ?", iid)
            raise UserError(f"{name}: görüntü okunamadı ({type(e).__name__}: {e})") from e
        self.db.run("UPDATE images SET file = ?, display = ?, width = ?, height = ? WHERE id = ?",
                    str(path.relative_to(self.s.data_dir)), str(display.relative_to(self.s.data_dir)), w, h, iid)
        img = self.db.one("SELECT * FROM images WHERE id = ?", iid)
        try:
            self.ensure_task(ds, img)
        except LSError as e:                                 # görüntü durur; "Label Studio'ya eşitle" ile tekrar denenir
            img["ls_error"] = str(e)
        return img

    def path(self, img: dict, key: str = "file") -> Path:
        return self.s.data_dir / img[key]

    def ensure_task(self, ds: dict, img: dict) -> int:
        if img.get("ls_task"):
            return img["ls_task"]
        tid = self.ls().upload_image(ds["ls_project"], self.path(img, "display"))
        self.db.run("UPDATE images SET ls_task = ? WHERE id = ?", tid, img["id"])
        img["ls_task"] = tid
        return tid

    def sync_tasks(self, ds: dict) -> int:
        n = 0
        for img in self.db.images(ds["id"]):
            if not img["ls_task"]:
                self.ensure_task(ds, img)
                n += 1
        return n

    def delete_image(self, ds: dict, img: dict) -> None:
        if img["ls_task"]:
            try:
                self.ls().delete_task(img["ls_task"])
            except LSError as e:
                if "404" not in str(e):
                    raise
        shutil.rmtree(self.image_dir(img), ignore_errors=True)
        self.db.run("DELETE FROM images WHERE id = ?", img["id"])

    # -------------------------------------------------------------- Label Studio durumu
    def ls_tasks(self, ds: dict) -> dict[int, object]:
        return {t.id: t for t in self.ls().tasks(ds["ls_project"])}

    def task_status(self, ds: dict) -> dict[int, dict]:
        """görev no -> {labeled, versions, annotations}"""
        out = {}
        for tid, t in self.ls_tasks(ds).items():
            anns = [a for a in (t.annotations or []) if not _get(a, "was_cancelled")]
            out[tid] = {"labeled": bool(anns), "annotations": len(anns),
                        "versions": sorted({_get(p, "model_version") for p in (t.predictions or [])} - {None})}
        return out

    # -------------------------------------------------------------- çalıştırıcı sonuçları
    def store_result(self, job: dict, img: dict, payload: dict) -> None:
        """Çalıştırıcının bir görüntü için gönderdiği tahmini Label Studio'ya ön etiket olarak ekler."""
        if payload.get("error"):
            self.db.run("UPDATE job_items SET status = 'error', error = ? WHERE job_id = ? AND image_id = ?",
                        str(payload["error"])[:2000], job["id"], img["id"])
            return
        pred = payload["prediction"]
        size = convert.result_size(pred["result"])
        if size and img["height"] and size != (img["height"], img["width"]):
            raise UserError(f"tahmin boyutu {size[1]}x{size[0]}, görüntü {img['width']}x{img['height']}")
        ds = self.db.dataset(img["dataset_id"])
        tid = self.ensure_task(ds, img)
        ls = self.ls()
        ls.add_prediction(tid, pred)
        if pred["model_version"] != job["model_version"]:
            self.db.run("UPDATE jobs SET model_version = ? WHERE id = ?", pred["model_version"], job["id"])
            job["model_version"] = pred["model_version"]
        first = not self.db.one("SELECT 1 FROM job_items WHERE job_id = ? AND status = 'done'", job["id"])
        self.db.run("UPDATE job_items SET status = 'done', summary = ?, seconds = ?, error = NULL "
                    "WHERE job_id = ? AND image_id = ?", json.dumps(payload.get("summary") or {}),
                    payload.get("seconds"), job["id"], img["id"])
        if first:                                           # projede gösterilecek ön etiket sürümü
            ls.set_model_version(ds["ls_project"], pred["model_version"])

    # -------------------------------------------------------------- karşılaştırma
    def _pairs(self, ds: dict):
        dims = {i["ls_task"]: (i["height"], i["width"]) for i in self.db.images(ds["id"]) if i["ls_task"]}
        for t in self.ls_tasks(ds).values():
            ann = latest_annotation(t)
            if ann is not None:
                yield t, ann, dims.get(t.id)

    def model_versions(self, ds: dict) -> list[str]:
        vs = set()
        for st in self.task_status(ds).values():
            vs.update(st["versions"])
        return sorted(vs)

    def compare(self, ds: dict, version: str | None) -> tuple[dict, list[dict]]:
        rows = []
        by_task = {i["ls_task"]: i for i in self.db.images(ds["id"])}
        for task, ann, dims in self._pairs(ds):
            r = core.compare_task(task, ann, core.task_shape(task, ann, dims), self.s.classes, version)
            if r is not None:
                img = by_task.get(task.id)
                r["image_id"] = img["id"] if img else None
                if img:
                    r["image"] = img["name"]
                rows.append(r)
        if not rows:
            raise UserError("karşılaştırılacak görev yok: hem model tahmini hem insan etiketi olan görev gerekli "
                            "(Label Studio'da en az bir görevi gönderin)")
        rows.sort(key=lambda r: r["image"])
        return evaluate.summarize(rows), rows

    def save_report(self, ds: dict, user_id: int, version: str | None) -> int:
        summary, rows = self.compare(ds, version)
        return self.db.run("INSERT INTO reports (dataset_id, user_id, model_version, summary, rows, created) "
                           "VALUES (?, ?, ?, ?, ?, ?)", ds["id"], user_id, version, json.dumps(summary),
                           json.dumps(rows), now())

    def overlay(self, ds: dict, img: dict, version: str | None) -> bytes:
        """Görüntü üstüne model ve insan etiketleri. Hücre sınırı: model eflatun, insan yeşil. Nörit: ikisi de sarı,
        yalnız model kırmızı, yalnız insan camgöbeği. Grup sınırı: model turuncu, insan mavi."""
        t = self.ls().task(img["ls_task"])
        ann, pred = latest_annotation(t), prediction_for(t, version)
        shape = (img["height"], img["width"])
        empty = {"cell_labels": np.zeros(shape, np.uint16), "branch": np.zeros(shape, bool), "group": np.zeros(shape, bool)}
        m = convert.to_masks(pred["result"], shape, self.s.classes) if pred else empty
        h = convert.to_masks(ann["result"], shape, self.s.classes) if ann else empty
        with Image.open(self.path(img, "display")) as im:
            base = np.asarray(im.convert("RGB")).copy()
        f = min(1.0, OVERLAY_MAX / max(shape))
        if f < 1:
            size = (round(shape[1] * f), round(shape[0] * f))
            base = cv2.resize(base, size, interpolation=cv2.INTER_AREA)
            rs = lambda a: cv2.resize(a.astype(np.uint16 if a.dtype == np.uint16 else np.uint8), size,
                                      interpolation=cv2.INTER_NEAREST)
            m = {k: rs(v) if k == "cell_labels" else rs(v).astype(bool) for k, v in m.items()}
            h = {k: rs(v) if k == "cell_labels" else rs(v).astype(bool) for k, v in h.items()}
        out = base.astype(np.float32)
        for mask, col in ((m["branch"] & h["branch"], (255, 220, 0)), (m["branch"] & ~h["branch"], (255, 40, 40)),
                          (h["branch"] & ~m["branch"], (0, 220, 255))):
            out[mask] = out[mask] * 0.35 + np.float32(col) * 0.65
        out = out.astype(np.uint8)

        def contours(lab: np.ndarray, col, thick: int):
            for i in (int(v) for v in np.unique(lab) if v):
                cs, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(out, cs, -1, col, thick)

        contours(m["group"].astype(np.uint8), (255, 140, 0), 3)
        contours(h["group"].astype(np.uint8), (60, 110, 255), 2)
        contours(m["cell_labels"], (255, 0, 255), 3)
        contours(h["cell_labels"], (0, 255, 0), 1)
        buf = io.BytesIO()
        Image.fromarray(out).save(buf, "PNG", optimize=False)
        return buf.getvalue()

    # -------------------------------------------------------------- dışa aktarma
    def export_zip(self, ds: dict, formats: list[str]) -> Path:
        """Kontrol edilmiş görevlerin son insan etiketleri -> zip (maskeler, COCO, YOLO, ham Label Studio JSON)."""
        formats = [f for f in formats if f in export.FORMATS] or list(export.FORMATS)
        by_task = {i["ls_task"]: i for i in self.db.images(ds["id"])}
        pairs = list(self._pairs(ds))
        if not pairs:
            raise UserError("dışa aktarılacak kontrol edilmiş görev yok")
        tmp = Path(tempfile.mkdtemp(prefix="eng_export_"))
        out = tmp / "export"
        raw, coco = [], export.Coco(self.s.classes)
        for task, ann, dims in pairs:
            img = by_task.get(task.id)
            name = img["name"] if img else f"task_{task.id}.png"
            masks = convert.to_masks(ann["result"], core.task_shape(task, ann, dims), self.s.classes)
            stem = Path(name).stem
            raw.append({"task": task.id, "image": name, "annotation": ann})
            if "masks" in formats:
                export.write_masks(out / "masks", stem, masks)
            if "coco" in formats:
                coco.add(Path(img["display"]).name if img else name, masks)
            if "yolo" in formats:
                export.write_yolo(out / "yolo", stem, self.path(img, "display") if img else None, masks, self.s.classes)
        if "coco" in formats:
            coco.write(out / "coco.json")
        (out / "ls_tasks.json").write_text(json.dumps(raw, indent=1, default=str))
        z = tmp / f"{safe_name(ds['name'])}_export.zip"
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(out.rglob("*")):
                if p.is_file():
                    zf.write(p, p.relative_to(out))
        shutil.rmtree(out)
        return z


def _get(obj, key):
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def report_markdown(summary: dict, rows: list[dict]) -> str:
    return evaluate.to_markdown(summary, rows)
