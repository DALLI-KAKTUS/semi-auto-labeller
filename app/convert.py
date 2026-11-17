"""Maske <-> Label Studio sonuç JSON'u.

Hücre ve grup -> PolygonLabels (from_name "objects"), nörit -> BrushLabels (from_name "neurite", RLE). Koordinatlar
görüntü boyutuna göre yüzdedir; poligon noktaları piksel merkezleridir ((x + 0.5) / W * 100), geri dönüşte aynı
kural tersine uygulanır. EPS = 0.3 ile model maskesi -> poligon -> maske turu piksel piksel aynıdır (test resminde
ölçüldü; 0.7'de küçük hücrelerde IoU 0.97'ye düşüyor).
"""
from __future__ import annotations

import uuid

import cv2
import numpy as np
from label_studio_sdk.converter.brush import decode_rle, mask2rle

from .model import Prediction

POLY, BRUSH, IMAGE = "objects", "neurite", "image"
EPS = 0.3                          # poligon sadeleştirme (piksel): yalnız doğrusal ara noktaları atar


def _rid() -> str:
    return uuid.uuid4().hex[:10]


def mask_polygons(mask: np.ndarray, largest_only: bool = False) -> list[np.ndarray]:
    """bool maske -> dış konturlar (N x 2, piksel, x/y). Delikler yok sayılır."""
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cs = sorted(cs, key=cv2.contourArea, reverse=True)[:1] if largest_only else cs
    out = []
    for c in cs:
        p = cv2.approxPolyDP(c, EPS, True)[:, 0, :]
        if len(p) >= 3:
            out.append(p.astype(np.float64))
    return out


def _poly_item(pts: np.ndarray, label: str, w: int, h: int, score: float | None = None) -> dict:
    pct = np.column_stack([(pts[:, 0] + 0.5) / w * 100, (pts[:, 1] + 0.5) / h * 100])
    item = {"id": _rid(), "from_name": POLY, "to_name": IMAGE, "type": "polygonlabels",
            "original_width": w, "original_height": h, "image_rotation": 0,
            "value": {"points": np.round(pct, 4).tolist(), "closed": True, "polygonlabels": [label]}}
    if score is not None:
        item["score"] = round(float(score), 4)
    return item


def _brush_item(mask: np.ndarray, label: str) -> dict:
    h, w = mask.shape
    return {"id": _rid(), "from_name": BRUSH, "to_name": IMAGE, "type": "brushlabels",
            "original_width": w, "original_height": h, "image_rotation": 0,
            "value": {"format": "rle", "rle": mask2rle(mask.astype(np.uint8) * 255), "brushlabels": [label]}}


def to_ls(pred: Prediction, classes: dict, min_score: float = 0.0) -> dict:
    """Prediction -> Label Studio prediction ({model_version, score, result}) + özet sayılar."""
    h, w = pred.shape
    result, kept = [], []
    for i, s in sorted(pred.cell_scores.items()):
        if s < min_score:
            continue
        for pts in mask_polygons(pred.cell_labels == i, largest_only=True):
            result.append(_poly_item(pts, classes["cell"], w, h, s))
            kept.append(s)
    n_group = 0
    if pred.group.any():
        for pts in mask_polygons(pred.group):
            result.append(_poly_item(pts, classes["group"], w, h))
            n_group += 1
    if pred.branch.any():
        result.append(_brush_item(pred.branch, classes["branch"]))
    score = float(np.mean(kept)) if kept else 0.0
    return {"model_version": pred.model_version, "score": round(score, 4), "result": result,
            "summary": {"cells": len(kept), "cells_below_threshold": len(pred.cell_scores) - len(kept),
                        "groups": n_group, "neurite_px": int(pred.branch.sum()), "width": w, "height": h}}


# ------------------------------------------------------------------ geri dönüşüm
def result_size(result: list[dict]) -> tuple[int, int] | None:
    for r in result:
        if "original_width" in r and "original_height" in r:
            return int(r["original_height"]), int(r["original_width"])
    return None


def _fill(points_pct, h: int, w: int) -> np.ndarray:
    p = np.asarray(points_pct, np.float64)
    px = np.column_stack([p[:, 0] / 100 * w - 0.5, p[:, 1] / 100 * h - 0.5])
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.round(px * 16).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=4)
    return m.astype(bool)


def to_masks(result: list[dict], shape: tuple[int, int], classes: dict) -> dict[str, np.ndarray]:
    """Label Studio sonuçları (tahmin ya da insan etiketi) -> cell_labels (uint16), branch, group (bool)."""
    h, w = shape
    cells = np.zeros((h, w), np.uint16)
    branch = np.zeros((h, w), bool)
    group = np.zeros((h, w), bool)
    n = 0
    for r in result:
        v = r.get("value", {})
        if r.get("type") == "polygonlabels":
            labels = v.get("polygonlabels", [])
            m = _fill(v["points"], h, w)
            if classes["cell"] in labels:
                n += 1
                cells[m] = n
            elif classes["group"] in labels:
                group |= m
        elif r.get("type") == "brushlabels" and classes["branch"] in v.get("brushlabels", []) and v.get("rle"):
            branch |= decode_rle(v["rle"]).reshape(h, w, 4)[..., 3] > 0
    return {"cell_labels": cells, "branch": branch, "group": group}
