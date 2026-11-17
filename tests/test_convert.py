"""Model gerektirmeyen testler: maske <-> Label Studio JSON turu, karşılaştırma, COCO RLE.  python -m pytest tests"""
from pathlib import Path

import numpy as np

from app import convert, evaluate, export
from app.model import Prediction

CLASSES = {"cell": "Cell", "branch": "Neurite", "group": "Group"}


def _sample():
    h, w = 120, 160
    yy, xx = np.mgrid[:h, :w]
    cells = np.zeros((h, w), np.uint16)
    cells[(yy - 30) ** 2 + (xx - 40) ** 2 < 100] = 1
    cells[(yy - 80) ** 2 + (xx - 110) ** 2 < 49] = 2
    cells[5:9, 140:146] = 3
    branch = np.zeros((h, w), bool)
    branch[30, 50:120] = True
    branch[31:80, 110] = True
    group = np.zeros((h, w), bool)
    group[90:115, 10:60] = True
    return Prediction(Path("a.png"), Path("a.png"), cells, {1: 0.9, 2: 0.4, 3: 0.8}, branch, group, "v")


def test_round_trip_is_pixel_exact():
    p = _sample()
    ls = convert.to_ls(p, CLASSES)
    back = convert.to_masks(ls["result"], p.shape, CLASSES)
    assert ls["summary"]["cells"] == 3 and ls["summary"]["groups"] == 1
    for i in (1, 2, 3):
        assert ((back["cell_labels"] == i) == (p.cell_labels == i)).all()
    assert (back["branch"] == p.branch).all() and (back["group"] == p.group).all()


def test_min_score_drops_cells():
    ls = convert.to_ls(_sample(), CLASSES, min_score=0.5)
    assert ls["summary"]["cells"] == 2 and ls["summary"]["cells_below_threshold"] == 1


def test_compare_counts_edits():
    p = _sample()
    model = {"cell_labels": p.cell_labels, "branch": p.branch, "group": p.group}
    human = {k: v.copy() for k, v in model.items()}
    human["cell_labels"][human["cell_labels"] == 2] = 0           # silinen
    human["cell_labels"][60:70, 10:20] = 4                         # eklenen
    yy, xx = np.mgrid[:120, :160]
    human["cell_labels"][(human["cell_labels"] == 0) & (np.abs(yy - 30) < 11) & (np.abs(xx - 40) < 11)] = 1  # büyütülen
    human["branch"][31:80, 110] = False
    r = evaluate.compare(model, human)
    assert (r["cells"]["deleted"], r["cells"]["added"], r["cells"]["edited"], r["cells"]["unchanged"]) == (1, 1, 1, 1)
    assert r["neurite"]["removed_px"] == 49 and r["neurite"]["added_px"] == 0
    s = evaluate.summarize([r])
    assert s["cells"]["matched"] == 2 and s["group"]["dice"] == 1.0


def test_coco_rle_decodes():
    m = _sample().branch
    rle = export._rle(m)
    assert sum(rle["counts"]) == m.size
    flat = np.zeros(m.size, np.uint8)
    pos = 0
    for i, k in enumerate(rle["counts"]):
        flat[pos:pos + k] = i % 2
        pos += k
    assert (flat.reshape(m.shape[::-1]).T.astype(bool) == m).all()
