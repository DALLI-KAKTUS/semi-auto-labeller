"""Web uygulaması + yerel çalıştırıcı, sahte Label Studio ile uçtan uca (model gerektirmez: Python eklenti modeli)."""
from __future__ import annotations

import copy
import io
import json
import os
import zipfile
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote

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

    def task_summaries(self, pid):
        return [{"id": t.id, "annotations": sum(1 for a in t.annotations if not a.get("was_cancelled")),
                 "versions": sorted({p["model_version"] for p in t.predictions})} for t in self.tasks(pid)]

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


def test_help_requires_login_and_shows_team_workflow(env):
    anon = TestClient(env.app)
    r = anon.get("/help", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].startswith("/login")
    r = env.client.get("/help")
    assert r.status_code == 200
    for text in ("Ekip kullanım rehberi", "Yerel çalıştırıcılar", "Submit", "Rapor oluştur", "henüz harici yedek yok"):
        assert text in r.text
    assert 'href="/help"' in env.client.get("/").text


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

    # çalıştırıcı paketi: başlatıcılar ve sunucu adresi var, config.yaml (anahtar) yok
    zf = zipfile.ZipFile(io.BytesIO(c.get("/download/runner.zip").content))
    z = zf.namelist()
    assert "eng-calistirici/runner/__main__.py" in z and "eng-calistirici/models/lite_v8.json" in z
    assert "eng-calistirici/requirements-runner.txt" in z and "eng-calistirici/modeller/BENIOKU.txt" in z
    assert not any(n.endswith("config.yaml") for n in z)
    assert json.loads(zf.read("eng-calistirici/runner/server.json"))["server"] == "http://testserver"
    assert (zf.getinfo("eng-calistirici/ENG-Calistirici.command").external_attr >> 16) & 0o111
    assert b"\r\n" in zf.read("eng-calistirici/ENG-Calistirici.bat")

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


def test_pairing_connects_runner_without_copying_token(env, tmp_path, monkeypatch):
    """Çalıştırıcı kod ister, kullanıcı web'de onaylar, çalıştırıcı anahtarı alır ve kaydeder; kod tek kullanımlık."""
    from runner import __main__ as R

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    anon = TestClient(env.app)
    shown = {}

    def approve(_seconds):
        if "approved" not in shown:
            # oturumsuz açılan onay sayfası girişe yönlendirir ve kodu kaybetmez
            r = anon.get(f"/baglan?kod={shown['code']}", follow_redirects=False)
            assert "next=/baglan%3Fkod%3D" in r.headers["location"]
            page = env.client.get(f"/baglan?kod={shown['code'].lower().replace('-', ' ')}").text
            assert shown["code"] in page and "Onayla" in page
            r = env.client.post("/baglan", data={"code": shown["code"], "action": "approve", "name": "lab pc"},
                                follow_redirects=False)
            assert "msg=" in r.headers["location"]
            shown["approved"] = True

    got = R.pair("http://testserver", http=anon, show=shown.update, sleep=approve)
    assert got["status"] == "approved" and got["token"].startswith("engr_") and got["name"] == "lab pc"
    assert shown["url"] == f"http://testserver/baglan?kod={shown['code']}"
    runner = env.app.state.db.one("SELECT * FROM runners")
    assert runner["name"] == "lab pc" and json.loads(runner["info"])["hostname"]
    # aynı gizli anahtarla ikinci kez anahtar alınamaz; kod artık geçersiz
    assert anon.post("/api/runner/pair/poll", json={"secret": shown["secret"]}).json()["status"] == "expired"
    assert "süresi dolmuş" in env.client.get(f"/baglan?kod={shown['code']}").text
    rc = TestClient(env.app, headers={"Authorization": f"Bearer {got['token']}"})
    assert rc.post("/api/runner/poll", json={"models": [], "wait": 0}).status_code == 200

    # reddedilen istek anahtar vermez
    p = anon.post("/api/runner/pair", json={"info": {"hostname": "yabanci"}}).json()
    assert anon.post("/api/runner/pair/poll", json={"secret": p["secret"]}).json()["status"] == "pending"
    env.client.post("/baglan", data={"code": p["code"], "action": "deny"})
    assert anon.post("/api/runner/pair/poll", json={"secret": p["secret"]}).json() == {"status": "denied"}
    assert len(env.app.state.db.all("SELECT id FROM runners")) == 1

    # ayarlar yalnız kullanıcının okuyabileceği dosyaya yazılır
    R.save_settings({"servers": {"http://testserver": {"token": got["token"]}}, "models": []})
    assert R.load_settings()["servers"]["http://testserver"]["token"] == got["token"]
    if os.name != "nt":
        assert R.config_path().stat().st_mode & 0o077 == 0


def test_runner_finds_models_in_folder(tmp_path):
    from runner.__main__ import discover_models, model_spec

    cards, folder = tmp_path / "models", tmp_path / "modeller"
    cards.mkdir()
    folder.mkdir()
    (cards / "lite_v8.json").write_text(json.dumps({"name": "lite_v8", "weights": "lite_net.pt", "format": "smp"}))
    (folder / "lite_net.pt").write_bytes(b"x")
    (folder / "own.onnx").write_bytes(b"x")
    (folder / "own.json").write_text(json.dumps({"name": "own", "weights": "own.onnx", "format": "onnx"}))
    (folder / "other.pt").write_bytes(b"x")
    (folder / "BENIOKU.txt").write_text("-")
    specs, notes = discover_models(folder, cards)
    assert specs == [f"{folder / 'own.json'}={folder / 'own.onnx'}", f"{cards / 'lite_v8.json'}={folder / 'lite_net.pt'}"]
    assert len(notes) == 1 and "other.pt" in notes[0] and "lite_net.pt" in notes[0]
    # pencereye sürüklenen yol (tırnaklı / kaçışlı boşluk), kart adından ya da sorularak bulunur
    spaced = tmp_path / "my models" / "lite_net.pt"
    spaced.parent.mkdir()
    spaced.write_bytes(b"x")
    assert model_spec(f"'{spaced}' ", cards) == f"{cards / 'lite_v8.json'}={spaced}"
    if os.name != "nt":
        assert model_spec(str(spaced).replace(" ", "\\ "), cards) == f"{cards / 'lite_v8.json'}={spaced}"
    assert model_spec(str(folder / "other.pt"), cards, choose=lambda cs: cs[0]) == f"{cards / 'lite_v8.json'}={folder / 'other.pt'}"
    with pytest.raises(Exception):
        model_spec(str(folder / "other.pt"), cards)


def test_torchscript_upload_rejected(env):
    r = env.client.post("/datasets", data={"name": "ts"}, follow_redirects=False)
    did = int(r.headers["location"].split("/")[2].split("?")[0])
    card = {"name": "ts", "format": "torchscript"}
    r = env.client.post(f"/datasets/{did}/models", files={
        "card": ("card.json", json.dumps(card).encode()), "weights": ("m.pt", b"x")}, follow_redirects=False)
    assert "err=" in r.headers["location"] and "TorchScript" in unquote(r.headers["location"])


def test_runner_without_local_model_accepts_web_models(tmp_path, monkeypatch):
    from app import config as C
    monkeypatch.setattr(C, "load", lambda: {"model": {"card": str(tmp_path / "no_model.json"), "device": "cpu"}})
    assert parse_models([], "cpu") == {}


def test_private_runner_only_owner_can_use(env):
    """Paylaşılmayan çalıştırıcıya yalnız sahibi iş verebilir; sahibi paylaşınca herkes."""
    from web import auth

    db = env.app.state.db
    auth.create_user(db, "ayse", "ayse-pass-1")
    other = TestClient(env.app)
    other.post("/login", data={"username": "ayse", "password": "ayse-pass-1"})
    r = env.client.post("/datasets", data={"name": "ozel"}, follow_redirects=False)
    did = int(r.headers["location"].split("/")[2].split("?")[0])
    rid, _ = auth.create_runner(db, 1, "admin pc")
    db.run("UPDATE runners SET models = ? WHERE id = ?", json.dumps([{"name": "m", "version": "m-1"}]), rid)
    db.run("INSERT INTO images (dataset_id, name, file, created) VALUES (?, 'a.png', '', 'x')", did)

    assert "admin pc" not in other.get(f"/datasets/{did}").text
    r = other.post(f"/datasets/{did}/jobs", data={"model": f"{rid}|m"}, follow_redirects=False)
    assert "yaln%C4%B1z%20sahibi" in r.headers["location"] and not db.all("SELECT id FROM jobs")
    # sahibi olmayan paylaşımı değiştiremez
    other.post(f"/runners/{rid}/share", data={"shared": "1"})
    assert db.one("SELECT shared FROM runners WHERE id = ?", rid)["shared"] == 0
    env.client.post(f"/runners/{rid}/share", data={"shared": "1"})
    assert "admin pc" in other.get(f"/datasets/{did}").text
    r = other.post(f"/datasets/{did}/jobs", data={"model": f"{rid}|m"}, follow_redirects=False)
    assert "err=" not in r.headers["location"] and len(db.all("SELECT id FROM jobs")) == 1
    env.client.post(f"/runners/{rid}/share")
    assert db.one("SELECT shared FROM runners WHERE id = ?", rid)["shared"] == 0


def test_pairing_shared_flag(env):
    anon = TestClient(env.app)
    for shared, want in (("1", 1), ("", 0)):
        p = anon.post("/api/runner/pair", json={"info": {"hostname": "h"}}).json()
        env.client.post("/baglan", data={"code": p["code"], "action": "approve", "name": f"pc{want}", "shared": shared})
        rid = anon.post("/api/runner/pair/poll", json={"secret": p["secret"]}).json()["runner_id"]
        assert env.app.state.db.one("SELECT shared FROM runners WHERE id = ?", rid)["shared"] == want


def test_copy_dataset_copies_images_without_labels(env):
    db = env.app.state.db
    r = env.client.post("/datasets", data={"name": "kaynak"}, follow_redirects=False)
    did = int(r.headers["location"].split("/")[2].split("?")[0])
    buf = io.BytesIO()
    Image.new("L", (80, 60), 128).save(buf, "PNG")
    assert env.client.post(f"/api/datasets/{did}/images", files={"file": ("a.png", buf.getvalue())}).status_code == 200
    src = db.images(did)[0]
    env.ls.tasks_[src["ls_task"]].annotations.append({"result": [], "was_cancelled": False})
    r = env.client.post(f"/datasets/{did}/copy", data={"name": "kopya"}, follow_redirects=False)
    new = int(r.headers["location"].split("/")[2].split("?")[0])
    assert new != did and db.dataset(new)["ls_project"] != db.dataset(did)["ls_project"]
    img = db.images(new)[0]
    assert img["name"] == "a.png" and img["ls_task"] and img["ls_task"] != src["ls_task"]
    # dosya yeniden yüklenmedi: aynı dosyaya sabit bağlantı
    a, b = env.settings.data_dir / src["file"], env.settings.data_dir / img["file"]
    assert a != b and a.read_bytes() == b.read_bytes() and (os.name == "nt" or a.stat().st_ino == b.stat().st_ino)
    assert not env.ls.tasks_[img["ls_task"]].annotations and env.ls.tasks_[img["ls_task"]].predictions == []
    page = env.client.get(f"/datasets/{new}").text
    assert "a.png" in page and "kopya" in page


def test_server_runner_needs_admin_permission(tmp_path):
    from web import auth

    ls = FakeLS()
    key = tmp_path / "key" / "token"
    s = Settings(data_dir=tmp_path / "data", ls_url="http://ls:8080", ls_api_key="x", admin_username="admin",
                 admin_password="admin-pass-1", server_runner_token_file=str(key))
    app = create_app(s, ls_factory=lambda: ls)
    db = app.state.db
    srv = db.one("SELECT * FROM runners WHERE kind = 'server'")
    token = key.read_text()
    assert srv and srv["token_hash"] == auth.token_hash(token) and key.stat().st_mode & 0o077 == 0
    # yeniden başlatmada anahtar değişmez; dosya bozulursa yenilenir
    create_app(s, ls_factory=lambda: ls)
    assert key.read_text() == token
    key.write_text("bozuk")
    create_app(s, ls_factory=lambda: ls)
    assert key.read_text() != "bozuk" and len(db.all("SELECT id FROM runners WHERE kind = 'server'")) == 1
    from runner.__main__ import read_token_file
    assert read_token_file(key) == key.read_text()
    rc = TestClient(app, headers={"Authorization": f"Bearer {key.read_text()}"})
    rc.post("/api/runner/poll", json={"models": [{"name": "m", "version": "m-1", "device": "cpu"}], "wait": 0})

    admin = TestClient(app)
    admin.post("/login", data={"username": "admin", "password": "admin-pass-1"})
    auth.create_user(db, "ayse", "ayse-pass-1")
    uid = db.one("SELECT id FROM users WHERE username = 'ayse'")["id"]
    ayse = TestClient(app)
    ayse.post("/login", data={"username": "ayse", "password": "ayse-pass-1"})
    did = int(ayse.post("/datasets", data={"name": "d"}, follow_redirects=False).headers["location"].split("/")[2].split("?")[0])
    db.run("INSERT INTO images (dataset_id, name, file, created) VALUES (?, 'a.png', '', 'x')", did)
    rid = srv["id"]

    page = ayse.get(f"/datasets/{did}").text
    assert f'value="{rid}|m"' not in page and "izin isteyin" in page
    assert "yavaş" in ayse.get("/runners").text and "İzin iste" in ayse.get("/runners").text
    r = ayse.post(f"/datasets/{did}/jobs", data={"model": f"{rid}|m"}, follow_redirects=False)
    assert "err=" in r.headers["location"] and not db.all("SELECT id FROM jobs")
    # sunucu çalıştırıcısı silinemez, paylaşılamaz
    admin.post(f"/runners/{rid}/delete")
    assert db.one("SELECT id FROM runners WHERE id = ?", rid)

    ayse.post("/server-access", data={"note": "uzun iş"})
    assert "1 istek" in admin.get("/users").text and "uzun iş" in admin.get("/users").text
    assert ayse.post(f"/users/{uid}/server-access", data={"action": "approve"}).status_code == 403
    admin.post(f"/users/{uid}/server-access", data={"action": "approve"})
    assert f'value="{rid}|m"' in ayse.get(f"/datasets/{did}").text
    r = ayse.post(f"/datasets/{did}/jobs", data={"model": f"{rid}|m"}, follow_redirects=False)
    assert "err=" not in r.headers["location"] and len(db.all("SELECT id FROM jobs")) == 1
    admin.post(f"/users/{uid}/server-access", data={"action": "revoke"})
    r = ayse.post(f"/datasets/{did}/jobs", data={"model": f"{rid}|m", "scope": "all"}, follow_redirects=False)
    assert "err=" in r.headers["location"]
