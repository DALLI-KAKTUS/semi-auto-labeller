"""Düzeltilmiş etiketleri dosyaya yazma: maske PNG, COCO (instance segmentation), YOLO-seg."""
from __future__ import annotations

import datetime as dt
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .convert import mask_polygons

FORMATS = ("masks", "coco", "yolo")


def write_masks(out: Path, stem: str, m: dict) -> None:
    """<stem>_cells.png (16 bit, hücre no), <stem>_neurite.png ve <stem>_group.png (0/255)."""
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(m["cell_labels"].astype(np.uint16)).save(out / f"{stem}_cells.png")
    Image.fromarray(m["branch"].astype(np.uint8) * 255).save(out / f"{stem}_neurite.png")
    Image.fromarray(m["group"].astype(np.uint8) * 255).save(out / f"{stem}_group.png")


def _rle(mask: np.ndarray) -> dict:
    """COCO sıkıştırılmamış RLE (sütun öncelikli, sıfırla başlar)."""
    flat = mask.astype(np.uint8).ravel(order="F")
    bounds = np.concatenate([[0], np.flatnonzero(flat[1:] != flat[:-1]) + 1, [flat.size]])
    counts = np.diff(bounds).tolist()
    if flat[0] == 1:
        counts = [0] + counts
    return {"size": list(mask.shape), "counts": [int(c) for c in counts]}


def _bbox(mask: np.ndarray) -> list[float]:
    ys, xs = np.nonzero(mask)
    return [float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)]


class Coco:
    CATS = [(1, "cell"), (2, "group"), (3, "neurite")]

    def __init__(self, classes: dict):
        self.data = {"info": {"description": "ENG label app export", "date_created": dt.datetime.now().isoformat()},
                     "images": [], "annotations": [],
                     "categories": [{"id": i, "name": classes[k if k != "neurite" else "branch"]} for i, k in self.CATS]}

    def add(self, file_name: str, m: dict) -> None:
        h, w = m["cell_labels"].shape
        img_id = len(self.data["images"]) + 1
        self.data["images"].append({"id": img_id, "file_name": file_name, "width": w, "height": h})

        def ann(cat: int, mask: np.ndarray, seg, crowd: int = 0):
            self.data["annotations"].append({"id": len(self.data["annotations"]) + 1, "image_id": img_id,
                                             "category_id": cat, "segmentation": seg, "area": int(mask.sum()),
                                             "bbox": _bbox(mask), "iscrowd": crowd})

        for i in (int(v) for v in np.unique(m["cell_labels"]) if v):
            mask = m["cell_labels"] == i
            polys = mask_polygons(mask, largest_only=True)
            if polys:
                ann(1, mask, [p.ravel().tolist() for p in polys])
        for p in mask_polygons(m["group"]):
            mask = np.zeros_like(m["group"], np.uint8)
            cv2.fillPoly(mask, [p.astype(np.int32)], 1)
            ann(2, mask.astype(bool), [p.ravel().tolist()])
        if m["branch"].any():
            ann(3, m["branch"], _rle(m["branch"]), crowd=1)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.data))


def write_yolo(out: Path, stem: str, image: Path | None, m: dict, classes: dict) -> None:
    """YOLO-seg: sınıf 0 hücre, 1 grup (normalize poligonlar). Nörit ince yapı olduğu için YOLO'ya yazılmaz."""
    (out / "labels").mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(parents=True, exist_ok=True)
    h, w = m["cell_labels"].shape
    lines = []

    def line(cls: int, p: np.ndarray):
        xy = np.column_stack([np.clip(p[:, 0] / w, 0, 1), np.clip(p[:, 1] / h, 0, 1)]).ravel()
        lines.append(f"{cls} " + " ".join(f"{v:.6f}" for v in xy))

    for i in (int(v) for v in np.unique(m["cell_labels"]) if v):
        for p in mask_polygons(m["cell_labels"] == i, largest_only=True):
            line(0, p)
    for p in mask_polygons(m["group"]):
        line(1, p)
    (out / "labels" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    if image is not None and image.exists():
        shutil.copyfile(image, out / "images" / image.name)
    (out / "data.yaml").write_text(f"path: {out}\ntrain: images\nval: images\nnames:\n  0: {classes['cell']}\n"
                                   f"  1: {classes['group']}\n")
