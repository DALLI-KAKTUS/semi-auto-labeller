"""Web uygulaması + yerel çalıştırıcı, sahte Label Studio ile uçtan uca (model gerektirmez: Python eklenti modeli)."""
from __future__ import annotations

import copy
import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import convert
from app.ls_client import LSError
from runner.__main__ import Runner, parse_models
from web.server import create_app
from web.settings import Settings

PLUGIN = '''
import numpy as np
from PIL import Image

class Disk:
    version = "disk-1"

    def predict(self, path):
        h, w = np.asarray(Image.open(path)).shape[:2]
        yy, xx = np.mgrid[:h, :w]
        cells = np.zeros((h, w), np.uint16)
        cells[(yy - 20) ** 2 + (xx - 20) ** 2 < 64] = 1
        cells[(yy - 40) ** 2 + (xx - 60) ** 2 < 81] = 2
        branch = np.zeros((h, w), bool)
        branch[50:52, 10:70] = True
        return {"cell_labels": cells, "branch": branch, "cell_scores": {1: 0.9, 2: 0.4}}
'''


class FakeLS:
    """LSClient'ın web uygulamasının kullandığı kısmı."""

    def __init__(self):
        self.projects, self.tasks_, self.n = {}, {}, 0

    def _id(self):
        self.n += 1
        return self.n

    def get_or_create_project(self, pid, title, label_config):
        pid = self._id()
        self.projects[pid] = {"title": title, "model_version": None}
        return pid

    def delete_project(self, pid):
        self.projects.pop(pid)

    def upload_image(self, pid, path: Path):
        tid = self._id()
        self.tasks_[tid] = SimpleNamespace(id=tid, project=pid, data={"image": f"/data/upload/1/abcdef12-{path.name}"},
                                           annotations=[], predictions=[], total_predictions=0)
        return tid

    def add_prediction(self, tid, pred):
        t = self.tasks_[tid]
        t.predictions.append({"id": self._id(), "result": copy.deepcopy(pred["result"]),
                              "model_version": pred["model_version"], "score": pred["score"]})

    def set_model_version(self, pid, v):
        self.projects[pid]["model_version"] = v

    def tasks(self, pid):
        if pid not in self.projects:
            raise LSError("Label Studio'da bulunamadı (404)")
        return [t for t in self.tasks_.values() if t.project == pid]

    def task(self, tid):
        return self.tasks_[tid]

    def delete_task(self, tid):
        self.tasks_.pop(tid)


@pytest.fixture()
def env(tmp_path):
    ls = FakeLS()
    s = Settings(data_dir=tmp_path / "data", ls_url="http://ls:8080", ls_public_url="https://ls.example",
                 ls_api_key="x", admin_username="admin", admin_password="admin-pass-1")
    app = create_app(s, ls_factory=lambda: ls)
    client = TestClient(app)
    r = client.post("/login", data={"username": "admin", "password": "admin-pass-1"}, follow_redirects=False)
    assert r.status_code == 303 and "eng_session" in r.cookies
    return SimpleNamespace(ls=ls, client=client, app=app, tmp=tmp_path, settings=s)


def _png(tmp: Path, name: str, w=80, h=64, mode="RGB") -> Path:
    p = tmp / name
    a = (np.random.default_rng(0).random((h, w)) * 255).astype(np.uint8)
    Image.fromarray(np.stack([a] * 3, -1) if mode == "RGB" else (a.astype(np.uint16) * 200)).save(p)
    return p


def test_login_required(env):
    anon = TestClient(env.app)
    r = anon.get("/", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].startswith("/login")
    assert anon.post("/api/datasets/1/images").status_code == 401
    bad = anon.post("/login", data={"username": "admin", "password": "yanlis"}, follow_redirects=False)
    assert "err=" in bad.headers["location"]


def test_full_flow(env):
    c, ls = env.client, env.ls
    # veri seti = Label Studio projesi
    r = c.post("/datasets", data={"name": "deney 1"}, follow_redirects=False)
    did = int(r.headers["location"].split("/")[2].split("?")[0])
    ds_pid = next(iter(ls.projects))

    # görüntü yükleme: PNG doğrudan, 16 bit TIFF PNG'ye çevrilir; aynı ad ve bilinmeyen tür reddedilir
    png, tif = _png(env.tmp, "a.png"), _png(env.tmp, "b.tif", mode="I;16")
    for p in (png, tif):
        with open(p, "rb") as fh:
            r = c.post(f"/api/datasets/{did}/images", files={"file": (p.name, fh)})
        assert r.status_code == 200, r.text
    with open(png, "rb") as fh:
        assert c.post(f"/api/datasets/{did}/images", files={"file": ("a.png", fh)}).status_code == 400
    assert c.post(f"/api/datasets/{did}/images", files={"file": ("x.txt", b"hi")}).status_code == 400
    assert len(ls.tasks(ds_pid)) == 2
    assert all(t.data["image"].endswith(".png") for t in ls.tasks(ds_pid))       # TIFF -> PNG görevi

    # çalıştırıcı: anahtar oluştur, Python eklenti modeliyle bağlan
    page = c.post("/runners", data={"name": "test pc"}).text
    token = page.split('<pre class="copy">')[1].split("<")[0].strip()
    plugin = env.tmp / "disk_model.py"
    plugin.write_text(PLUGIN)
    runner = Runner("http://testserver", token, parse_models([f"python:{plugin}:Disk"], "cpu"))
    runner.http = TestClient(env.app, headers={"Authorization": f"Bearer {token}"})
    assert runner.poll(wait=0) is None                                           # iş yok, model ilan edildi
    assert "Disk" in c.get(f"/datasets/{did}").text

    # iş oluştur ve çalıştırıcı çalıştırsın
    rid = env.app.state.db.one("SELECT id FROM runners")["id"]
    r = c.post(f"/datasets/{did}/jobs", data={"model": f"{rid}|Disk", "scope": "missing", "min_score": "0.5"},
               follow_redirects=False)
    assert "msg=" in r.headers["location"], r.headers["location"]
    runner.loop(once=True)
    jobs = c.get(f"/api/datasets/{did}/jobs").json()["jobs"]
    assert jobs[0]["status"] == "done" and jobs[0]["done"] == 2, jobs
    assert ls.projects[ds_pid]["model_version"] == "disk-1"
    t1, t2 = ls.tasks(ds_pid)
    kinds = sorted(r["type"] for r in t1.predictions[0]["result"])
    assert kinds == ["brushlabels", "polygonlabels"]                             # güven eşiği 0.4'lük hücreyi attı

    # ikinci iş: kapsam "missing" -> hepsinin bu sürümde tahmini var
    r = c.post(f"/datasets/{did}/jobs", data={"model": f"{rid}|Disk", "scope": "missing"}, follow_redirects=False)
    assert "err=" in r.headers["location"]

    # insan etiketi: t1 aynen onaylandı, t2'de hücre silindi ve yeni hücre eklendi
    t1.annotations.append({"id": 101, "result": copy.deepcopy(t1.predictions[0]["result"]), "was_cancelled": False,
                           "updated_at": "2026-11-17T10:00:00"})
    h, w = 64, 80
    cells = np.zeros((h, w), np.uint16)
    cells[5:15, 50:65] = 1
    human = convert.to_ls(SimpleNamespace(shape=(h, w), cell_scores={1: 1.0}, cell_labels=cells,
                                          group=np.zeros((h, w), bool), branch=np.zeros((h, w), bool),
                                          model_version="human"), env.settings.classes)
    t2.annotations.append({"id": 102, "result": human["result"], "was_cancelled": False, "updated_at": "x"})

    page = c.get(f"/datasets/{did}").text
    assert "kontrol edildi" in page

    r = c.post(f"/datasets/{did}/compare", data={"version": "disk-1"}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].startswith("/reports/")
    rep = r.headers["location"]
    js = c.get(rep + "/report.json").json()
    s = js["summary"]
    assert s["tasks"] == 2 and s["tasks_unchanged"] == 1
    assert s["cells"]["model_cells"] == 2 and s["cells"]["human_cells"] == 2
    assert s["cells"]["matched"] == 1 and s["cells"]["deleted"] == 1 and s["cells"]["added"] == 1
    assert s["neurite"]["removed_px"] == 120 and s["neurite"]["added_px"] == 0
    html = c.get(rep).text
    assert "aynen onaylandı" in html and "bindirme" in html
    assert "# Model ve insan" in c.get(rep + "/report.md").text

    # bindirme görüntüsü
    iid = js["tasks"][0]["image_id"]
    r = c.get(f"/datasets/{did}/overlay/{iid}.png?version=disk-1")
    assert r.status_code == 200 and Image.open(io.BytesIO(r.content)).size == (80, 64)

    # dışa aktarma
    r = c.get(f"/datasets/{did}/export.zip?f=masks&f=coco")
    names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
    assert "coco.json" in names and "masks/a_cells.png" in names and "ls_tasks.json" in names
    coco = json.loads(zipfile.ZipFile(io.BytesIO(r.content)).read("coco.json"))
    assert len(coco["images"]) == 2

    # çalıştırıcı paketi config.yaml (anahtar) içermez
    z = zipfile.ZipFile(io.BytesIO(c.get("/download/runner.zip").content)).namelist()
    assert "eng-label-app/runner/__main__.py" in z and "eng-label-app/models/lite_v8.json" in z
    assert not any(n.endswith("config.yaml") for n in z)

    # silme: ad yanlışsa silinmez, doğruysa Label Studio projesiyle birlikte silinir
    r = c.post(f"/datasets/{did}/delete", data={"confirm": "yanlis", "delete_project": "1"}, follow_redirects=False)
    assert "err=" in r.headers["location"]
    c.post(f"/datasets/{did}/delete", data={"confirm": "deney 1", "delete_project": "1"})
    assert ds_pid not in ls.projects and not (env.settings.data_dir / "datasets" / str(did)).exists()


def test_runner_rejects_bad_token(env):
    r = TestClient(env.app).post("/api/runner/poll", json={}, headers={"Authorization": "Bearer yanlis"})
    assert r.status_code == 401


def test_cancel_job(env):
    c = env.client
    r = c.post("/datasets", data={"name": "d"}, follow_redirects=False)
    did = int(r.headers["location"].split("/")[2].split("?")[0])
    for n in ("a.png", "b.png"):
        with open(_png(env.tmp, n), "rb") as fh:
            c.post(f"/api/datasets/{did}/images", files={"file": (n, fh)})
    page = c.post("/runners", data={"name": "pc"}).text
    token = page.split('<pre class="copy">')[1].split("<")[0].strip()
    rc = TestClient(env.app, headers={"Authorization": f"Bearer {token}"})
    rc.post("/api/runner/poll", json={"models": [{"name": "m", "version": "v", "device": "cpu"}], "wait": 0})
    rid = env.app.state.db.one("SELECT id FROM runners")["id"]
    c.post(f"/datasets/{did}/jobs", data={"model": f"{rid}|m", "scope": "all"})
    job = rc.post("/api/runner/poll", json={"models": [], "wait": 0}).json()["job"]
    assert len(job["items"]) == 2
    c.post(f"/jobs/{job['id']}/cancel")
    r = rc.post(f"/api/runner/jobs/{job['id']}/items/{job['items'][0]['id']}", json={"error": "x"})
    assert r.json()["cancel"] is True
    assert rc.post("/api/runner/poll", json={"models": [], "wait": 0}).json()["job"] is None


def test_uploaded_onnx_runs_on_local_runner(env, monkeypatch):
    """Web'den model kartı + ağırlık yükle, çalıştırıcı indirip yerelde ONNX çalıştırsın; sunucu model çalıştırmasın."""
    import onnx
    from onnx import TensorProto, helper

    monkeypatch.setenv("HOME", str(env.tmp))
    graph = helper.make_graph([
        helper.make_node("Sub", ["x", "x"], ["zero"]),
        helper.make_node("Concat", ["zero", "x"], ["y"], axis=1),
    ], "demo", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, "h", "w"])],
       [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, "h", "w"])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    weights = env.tmp / "demo.onnx"
    onnx.save(model, weights)
    card = {"name": "demo", "format": "onnx", "weights": "wrong/path/ignored.onnx",
            "input": {"channel": "green", "percentile": None, "pad": {"multiple": 1}},
            "output": {"activation": "sigmoid", "class_channels": [0, 1], "classes": {"cell": 1}}}
    c = env.client
    r = c.post("/datasets", data={"name": "model testi"}, follow_redirects=False)
    did = int(r.headers["location"].split("/")[2].split("?")[0])
    with open(_png(env.tmp, "t.png"), "rb") as fh:
        c.post(f"/api/datasets/{did}/images", files={"file": ("t.png", fh)})
    r = c.post(f"/datasets/{did}/models", files={
        "card": ("card.json", json.dumps(card).encode()), "weights": ("demo.onnx", weights.read_bytes())
    }, follow_redirects=False)
    assert "msg=" in r.headers["location"], r.headers["location"]
    uploaded = env.app.state.db.one("SELECT * FROM models")
    assert uploaded["name"] == "demo" and uploaded["version"].startswith("demo-")
    assert "wrong/path" not in (env.settings.data_dir / uploaded["card"]).read_text()

    html = c.post("/runners", data={"name": "local"}).text
    token = html.split('<pre class="copy">')[1].split("<")[0].strip()
    rid = env.app.state.db.one("SELECT id FROM runners")["id"]
    runner = Runner("http://testserver", token, {})
    runner.http = TestClient(env.app, headers={"Authorization": f"Bearer {token}"})
    runner.poll(wait=0)
    r = c.post(f"/datasets/{did}/jobs", data={"model": f"{rid}|#{uploaded['id']}", "scope": "all"},
               follow_redirects=False)
    assert "msg=" in r.headers["location"]
    runner.loop(once=True)
    job = c.get(f"/api/datasets/{did}/jobs").json()["jobs"][0]
    assert job["status"] == "done" and job["done"] == 1
    assert env.ls.tasks(next(iter(env.ls.projects)))[0].predictions[0]["model_version"] == uploaded["version"]
    assert (env.tmp / ".cache" / "eng-label-app" / str(uploaded["id"]) / uploaded["version"] / "weights.onnx").exists()


def test_cross_origin_form_rejected(env):
    r = env.client.post("/datasets", data={"name": "csrf"}, headers={"Origin": "https://evil.example"})
    assert r.status_code == 403


def test_runner_without_local_model_accepts_web_models(tmp_path, monkeypatch):
    from app import config as C
    monkeypatch.setattr(C, "load", lambda: {"model": {"card": str(tmp_path / "no_model.json"), "device": "cpu"}})
    assert parse_models([], "cpu") == {}
