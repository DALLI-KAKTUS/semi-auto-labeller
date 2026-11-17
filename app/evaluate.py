"""Model ön etiketi ile insanın düzelttiği etiketin karşılaştırması.

Hücreler: IoU >= 0.5 ile bire bir eşleme (açgözlü, en yüksek IoU önce). Eşleşmeyen model hücresi = insanın sildiği,
eşleşmeyen insan hücresi = insanın eklediği; eşleşip IoU < 0.95 olanlar = sınırı düzeltilen. Nörit ve grup piksel
düzeyinde Dice ve IoU ile karşılaştırılır. İki taraf da aynı Label Studio JSON'undan aynı kuralla maskeye çevrilir.
"""
from __future__ import annotations

import numpy as np

MATCH_IOU = 0.5
EDIT_IOU = 0.95


def _overlaps(a: np.ndarray, b: np.ndarray) -> tuple[list[int], list[int], np.ndarray]:
    """İki örnek etiket görüntüsü -> (a numaraları, b numaraları, IoU matrisi)."""
    ia = [int(i) for i in np.unique(a) if i]
    ib = [int(i) for i in np.unique(b) if i]
    iou = np.zeros((len(ia), len(ib)))
    if not ia or not ib:
        return ia, ib, iou
    ra, rb = {v: k for k, v in enumerate(ia)}, {v: k for k, v in enumerate(ib)}
    both = (a > 0) & (b > 0)
    pairs, inter = np.unique(np.stack([a[both], b[both]]), axis=1, return_counts=True)
    area_a = {i: int((a == i).sum()) for i in ia}
    area_b = {i: int((b == i).sum()) for i in ib}
    for (x, y), n in zip(pairs.T, inter):
        iou[ra[int(x)], rb[int(y)]] = n / (area_a[int(x)] + area_b[int(y)] - n)
    return ia, ib, iou


def match_cells(model: np.ndarray, human: np.ndarray) -> dict:
    ia, ib, iou = _overlaps(model, human)
    matched, used_a, used_b = [], set(), set()
    for k in np.argsort(-iou, axis=None):
        r, c = np.unravel_index(k, iou.shape)
        if iou[r, c] < MATCH_IOU:
            break
        if r in used_a or c in used_b:
            continue
        used_a.add(r)
        used_b.add(c)
        matched.append(float(iou[r, c]))
    tp = len(matched)
    return {"model_cells": len(ia), "human_cells": len(ib), "matched": tp,
            "deleted": len(ia) - tp, "added": len(ib) - tp,
            "edited": sum(1 for v in matched if v < EDIT_IOU), "unchanged": sum(1 for v in matched if v >= EDIT_IOU),
            "mean_iou": round(float(np.mean(matched)), 4) if matched else None,
            "precision": round(tp / len(ia), 4) if ia else None, "recall": round(tp / len(ib), 4) if ib else None,
            "f1": round(2 * tp / (len(ia) + len(ib)), 4) if (ia or ib) else None}


def pixel_scores(model: np.ndarray, human: np.ndarray) -> dict:
    inter, a, b = int((model & human).sum()), int(model.sum()), int(human.sum())
    union = a + b - inter
    return {"model_px": a, "human_px": b, "added_px": int((human & ~model).sum()), "removed_px": int((model & ~human).sum()),
            "dice": round(2 * inter / (a + b), 4) if a + b else None, "iou": round(inter / union, 4) if union else None}


def compare(model: dict, human: dict) -> dict:
    return {"cells": match_cells(model["cell_labels"], human["cell_labels"]),
            "neurite": pixel_scores(model["branch"], human["branch"]),
            "group": pixel_scores(model["group"], human["group"])}


def summarize(rows: list[dict]) -> dict:
    """Görev başına karşılaştırmalar -> toplamlar (hücre sayıları toplanır, Dice/IoU piksel toplamlarından)."""
    cells = {k: sum(r["cells"][k] for r in rows) for k in ("model_cells", "human_cells", "matched", "deleted",
                                                          "added", "edited", "unchanged")}
    ious = [r["cells"]["mean_iou"] * r["cells"]["matched"] for r in rows if r["cells"]["mean_iou"] is not None]
    m, h, tp = cells["model_cells"], cells["human_cells"], cells["matched"]
    cells.update(mean_iou=round(sum(ious) / tp, 4) if tp else None, precision=round(tp / m, 4) if m else None,
                 recall=round(tp / h, 4) if h else None, f1=round(2 * tp / (m + h), 4) if m + h else None)
    out = {"tasks": len(rows), "tasks_unchanged": sum(1 for r in rows if r.get("unchanged")), "cells": cells}
    for key in ("neurite", "group"):
        a = sum(r[key]["model_px"] for r in rows)
        b = sum(r[key]["human_px"] for r in rows)
        removed = sum(r[key]["removed_px"] for r in rows)
        inter = a - removed
        out[key] = {"model_px": a, "human_px": b, "added_px": sum(r[key]["added_px"] for r in rows),
                    "removed_px": removed, "dice": round(2 * inter / (a + b), 4) if a + b else None,
                    "iou": round(inter / (a + b - inter), 4) if a + b - inter else None}
    return out


def to_markdown(summary: dict, rows: list[dict]) -> str:
    c, n, g = summary["cells"], summary["neurite"], summary["group"]
    lines = ["# Model ve insan etiketlerinin karşılaştırması", "",
             f"Kontrol edilen görev: {summary['tasks']} (hiç değiştirilmeden onaylanan: {summary['tasks_unchanged']})", "",
             "## Hücreler (IoU >= 0.5 eşleme)", "",
             "| model | insan | eşleşen | silinen | eklenen | sınırı düzeltilen | ort. IoU | kesinlik | duyarlılık | F1 |",
             "|---|---|---|---|---|---|---|---|---|---|",
             f"| {c['model_cells']} | {c['human_cells']} | {c['matched']} | {c['deleted']} | {c['added']} | {c['edited']} "
             f"| {c['mean_iou']} | {c['precision']} | {c['recall']} | {c['f1']} |", "",
             "## Nörit ve grup (piksel)", "",
             "| sınıf | model px | insan px | eklenen px | silinen px | Dice | IoU |", "|---|---|---|---|---|---|---|",
             f"| nörit | {n['model_px']} | {n['human_px']} | {n['added_px']} | {n['removed_px']} | {n['dice']} | {n['iou']} |",
             f"| grup | {g['model_px']} | {g['human_px']} | {g['added_px']} | {g['removed_px']} | {g['dice']} | {g['iou']} |",
             "", "## Görev başına", "", "| görüntü | model/insan hücre | silinen | eklenen | düzeltilen | nörit Dice |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        rc = r["cells"]
        lines.append(f"| {r['image']} | {rc['model_cells']}/{rc['human_cells']} | {rc['deleted']} | {rc['added']} "
                     f"| {rc['edited']} | {r['neurite']['dice']} |")
    return "\n".join(lines) + "\n"
