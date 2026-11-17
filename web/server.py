"""ENG label app web sunucusu (FastAPI).

    uvicorn web.server:app --host 0.0.0.0 --port 8000        (ayarlar: web/settings.py ortam değişkenleri)

Akış: kullanıcı veri seti açar (= Label Studio projesi) ve görüntülerini yükler (= görevler). Model kullanıcının kendi
bilgisayarındaki yerel çalıştırıcıda (python -m runner) çalışır: çalıştırıcı sunucuya dışarıdan bağlanıp iş sorar,
görüntüleri indirir, tahmin eder ve sonuçları gönderir; sunucu bunları Label Studio'ya ön etiket olarak ekler. İnsan
Label Studio'da düzeltir, bu uygulama model ile insan etiketlerini karşılaştırır ve dışa aktarır.
"""
from __future__ import annotations

import asyncio
import io
import json
import time
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.background import BackgroundTask
from urllib.parse import quote, urlsplit

from app import config as C, export
from app.ls_client import LSError

from . import auth
from .db import DB, JOB_ACTIVE, now
from .service import Service, UserError, report_markdown
from .settings import Settings

HERE = Path(__file__).resolve().parent
ONLINE_SECONDS = 60          # bu süre içinde soran çalıştırıcı çevrimiçi sayılır
POLL_SECONDS = 20            # uzun sorgu: iş yoksa çalıştırıcının isteği bu kadar bekletilir
PAIR_POLL_SECONDS = 2        # eşleştirmede çalıştırıcı onayı bu aralıkla sorar
PUBLIC = ("/login", "/static/", "/healthz", "/api/runner/")


class LoginRequired(Exception):
    pass


def create_app(settings: Settings | None = None, ls_factory=None) -> FastAPI:
    s = settings or Settings()
    db = DB(s.db_path)
    svc = Service(s, db, ls_factory)
    if s.admin_username and s.admin_password and not db.one("SELECT id FROM users"):
        auth.create_user(db, s.admin_username, s.admin_password, admin=True)

    app = FastAPI(title="ENG label app", docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    tpl = Jinja2Templates(directory=HERE / "templates")
    tpl.env.globals.update(ls_url=s.ls_public_url)
    tpl.env.filters["ago"] = _ago
    tpl.env.filters["pct"] = lambda v: "-" if v is None else f"{v * 100:.1f}%"
    tpl.env.filters["num"] = lambda v: "-" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)
    app.state.settings, app.state.db, app.state.svc = s, db, svc

    # ------------------------------------------------------------------ yardımcılar
    def user_of(request: Request) -> dict:
        u = auth.session_user(db, request.cookies.get(auth.SESSION_COOKIE))
        if not u:
            raise LoginRequired()
        return u

    def admin_of(request: Request) -> dict:
        u = user_of(request)
        if not u["admin"]:
            raise HTTPException(403, "yalnız yönetici")
        return u

    def dataset_or_404(did: int) -> dict:
        ds = db.dataset(did)
        if not ds:
            raise HTTPException(404, "veri seti yok")
        return ds

    def image_or_404(ds: dict, iid: int) -> dict:
        img = db.one("SELECT * FROM images WHERE id = ? AND dataset_id = ?", iid, ds["id"])
        if not img:
            raise HTTPException(404, "görüntü yok")
        return img

    def page(request: Request, name: str, user: dict, **ctx) -> HTMLResponse:
        return tpl.TemplateResponse(request, name, {"user": user, "msg": request.query_params.get("msg"),
                                                    "err": request.query_params.get("err"), **ctx})

    def back(url: str, msg: str | None = None, err: str | None = None) -> RedirectResponse:
        q = f"msg={quote(msg)}" if msg else f"err={quote(err)}" if err else ""
        return RedirectResponse(f"{url}{'&' if '?' in url else '?'}{q}" if q else url, status_code=303)

    def runners_view() -> list[dict]:
        rows = db.all("SELECT r.*, u.username AS owner FROM runners r JOIN users u ON u.id = r.user_id ORDER BY r.id")
        for r in rows:
            r["models"] = json.loads(r["models"] or "[]")
            r["info"] = json.loads(r["info"] or "{}")
            r["online"] = bool(r["last_seen"] and time.time() - r["last_seen"] < ONLINE_SECONDS)
        return rows

    def can_use(u: dict, r: dict) -> bool:
        """Çalıştırıcıya yalnız sahibi iş verebilir; sahibi paylaştıysa herkes (yönetici de istisna değil)."""
        return bool(r["shared"]) or r["user_id"] == u["id"]

    def usable_runners(u: dict) -> list[dict]:
        return [r for r in runners_view() if can_use(u, r)]

    def login_redirect(request: Request):
        if request.url.path.startswith("/api/"):
            return JSONResponse({"error": "oturum gerekli"}, status_code=401)
        target = request.url.path + (f"?{request.url.query}" if request.url.query else "")
        return RedirectResponse(f"/login?next={quote(target)}", status_code=303)

    @app.middleware("http")
    async def require_login(request: Request, call_next):
        """Varsayılan kapalı: giriş sayfası, statik dosyalar ve çalıştırıcı API'si (kendi anahtarı var) dışında her
        istek oturum ister (form doğrulamasından önce)."""
        p = request.url.path
        if not p.startswith(PUBLIC) and not auth.session_user(db, request.cookies.get(auth.SESSION_COOKIE)):
            return login_redirect(request)
        if request.method not in ("GET", "HEAD", "OPTIONS") and not p.startswith("/api/runner/"):
            origin = request.headers.get("origin") or request.headers.get("referer")
            if origin and urlsplit(origin).netloc != request.headers.get("host"):
                return JSONResponse({"error": "kaynak doğrulanamadı"}, status_code=403)
        return await call_next(request)

    @app.exception_handler(LoginRequired)
    async def _login_required(request: Request, _exc):
        return login_redirect(request)

    @app.exception_handler(UserError)
    async def _user_error(request: Request, exc: UserError):
        return JSONResponse({"error": str(exc)}, status_code=400)

    @app.exception_handler(LSError)
    async def _ls_error(request: Request, exc: LSError):
        return JSONResponse({"error": f"Label Studio: {exc}"}, status_code=502)

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    # ------------------------------------------------------------------ oturum
    @app.get("/login", response_class=HTMLResponse)
    def login_page(request: Request):
        return tpl.TemplateResponse(request, "login.html", {"err": request.query_params.get("err"),
                                                            "next": request.query_params.get("next", "/")})

    @app.post("/login")
    def login(username: str = Form(...), password: str = Form(...), next: str = Form("/")):
        token = auth.login(db, username, password)
        if not token:
            return back(f"/login?next={quote(next)}", err="kullanıcı adı ya da parola yanlış")
        r = RedirectResponse(next if next.startswith("/") and not next.startswith("//") else "/", status_code=303)
        r.set_cookie(auth.SESSION_COOKIE, token, max_age=auth.SESSION_DAYS * 86400, httponly=True, samesite="lax",
                     secure=s.secure_cookies)
        return r

    @app.post("/logout")
    def logout(request: Request):
        auth.logout(db, request.cookies.get(auth.SESSION_COOKIE))
        r = RedirectResponse("/login", status_code=303)
        r.delete_cookie(auth.SESSION_COOKIE)
        return r

    # ------------------------------------------------------------------ ana sayfa ve veri setleri
    @app.get("/", response_class=HTMLResponse)
    def home(request: Request):
        u = user_of(request)
        datasets = db.all(
            "SELECT d.*, u.username AS owner, (SELECT COUNT(*) FROM images i WHERE i.dataset_id = d.id) AS n_images "
            "FROM datasets d LEFT JOIN users u ON u.id = d.owner_id ORDER BY d.id DESC")
        return page(request, "home.html", u, datasets=datasets, runners=runners_view(), jobs=db.jobs(limit=10))

    @app.get("/help", response_class=HTMLResponse)
    def help_page(request: Request):
        return page(request, "help.html", user_of(request))

    @app.post("/datasets")
    def create_dataset(request: Request, name: str = Form(...)):
        u = user_of(request)
        try:
            did = svc.create_dataset(name, u["id"])
        except (UserError, LSError) as e:
            return back("/", err=str(e))
        return back(f"/datasets/{did}", msg="veri seti ve Label Studio projesi oluşturuldu")

    @app.get("/datasets/{did}", response_class=HTMLResponse)
    def dataset_page(request: Request, did: int):
        u = user_of(request)
        ds = dataset_or_404(did)
        images = db.images(did)
        ls_err, status = None, {}
        try:
            status = svc.task_status(ds)
        except Exception as e:                              # Label Studio kapalıysa sayfa yine açılsın
            ls_err = str(e)
        for img in images:
            st = status.get(img["ls_task"]) or {}
            img.update(labeled=st.get("labeled"), versions=st.get("versions", []),
                       link=svc.task_link(ds["ls_project"], img["ls_task"]))
        stats = {"total": len(images), "labeled": sum(1 for i in images if i["labeled"]),
                 "predicted": sum(1 for i in images if i["versions"]),
                 "missing_task": sum(1 for i in images if not i["ls_task"])}
        reports = db.all("SELECT id, model_version, summary, created FROM reports WHERE dataset_id = ? "
                         "ORDER BY id DESC LIMIT 10", did)
        for r in reports:
            r["summary"] = json.loads(r["summary"])
        return page(request, "dataset.html", u, ds=ds, images=images, stats=stats, ls_err=ls_err,
                    project_link=svc.project_link(ds["ls_project"]), runners=usable_runners(u), jobs=db.jobs(did, 20),
                    reports=reports, versions=sorted({v for i in images for v in i["versions"]}),
                    formats=export.FORMATS, accept=",".join(C.IMAGE_EXT),
                    uploaded_models=db.all("SELECT m.*, u.username AS owner FROM models m JOIN users u ON u.id = m.user_id ORDER BY m.id DESC"))

    @app.post("/datasets/{did}/models")
    def upload_model(request: Request, did: int, card: UploadFile = File(...), weights: UploadFile = File(...)):
        u = user_of(request)
        dataset_or_404(did)
        try:
            m = svc.add_model(u["id"], card, weights)
        except (UserError, ValueError) as e:
            return back(f"/datasets/{did}", err=str(e))
        return back(f"/datasets/{did}", msg=f"{m['name']} modeli yüklendi. Çalıştırıcı açıksa ön etiketleme başlatılabilir.")

    @app.post("/models/{mid}/delete")
    def delete_model(request: Request, mid: int):
        u = user_of(request)
        m = db.one("SELECT * FROM models WHERE id = ?", mid)
        if not m or not (u["admin"] or m["user_id"] == u["id"]):
            raise HTTPException(403)
        if db.one("SELECT id FROM jobs WHERE remote_model = ? AND status IN (?, ?)", mid, *JOB_ACTIVE):
            return back("/", err="aktif işteki model silinemez")
        db.run("DELETE FROM models WHERE id = ?", mid)
        _rmtree(s.data_dir / "models" / str(mid))
        return back("/", msg="model silindi")

    @app.post("/datasets/{did}/delete")
    def delete_dataset(request: Request, did: int, confirm: str = Form(""), delete_project: str = Form("")):
        u = user_of(request)
        ds = dataset_or_404(did)
        if not (u["admin"] or ds["owner_id"] == u["id"]):
            return back(f"/datasets/{did}", err="veri setini yalnız sahibi ya da yönetici silebilir")
        if confirm.strip() != ds["name"]:
            return back(f"/datasets/{did}", err="silmek için veri setinin adını aynen yazın")
        try:
            svc.delete_dataset(ds, bool(delete_project))
        except LSError as e:
            return back(f"/datasets/{did}", err=str(e))
        return back("/", msg=f"{ds['name']} silindi")

    @app.post("/api/datasets/{did}/images")
    def upload_image(request: Request, did: int, file: UploadFile = File(...)):
        user_of(request)
        ds = dataset_or_404(did)
        img = svc.add_image(ds, file.filename or "image", file.file)
        return {"id": img["id"], "name": img["name"], "ls_task": img["ls_task"], "warning": img.get("ls_error")}

    @app.post("/datasets/{did}/images/{iid}/delete")
    def delete_image(request: Request, did: int, iid: int):
        user_of(request)
        ds = dataset_or_404(did)
        img = image_or_404(ds, iid)
        try:
            svc.delete_image(ds, img)
        except LSError as e:
            return back(f"/datasets/{did}", err=str(e))
        return back(f"/datasets/{did}", msg=f"{img['name']} silindi")

    @app.post("/datasets/{did}/sync")
    def sync(request: Request, did: int):
        user_of(request)
        ds = dataset_or_404(did)
        try:
            n = svc.sync_tasks(ds)
        except LSError as e:
            return back(f"/datasets/{did}", err=str(e))
        return back(f"/datasets/{did}", msg=f"{n} görüntü Label Studio'ya yüklendi")

    @app.get("/datasets/{did}/thumb/{iid}.jpg")
    def thumb(request: Request, did: int, iid: int):
        user_of(request)
        img = image_or_404(dataset_or_404(did), iid)
        return FileResponse(svc.image_dir(img) / "thumb.jpg", headers={"Cache-Control": "max-age=86400"})

    # ------------------------------------------------------------------ işler (ön etiketleme)
    @app.post("/datasets/{did}/jobs")
    def create_job(request: Request, did: int, model: str = Form(...), scope: str = Form("missing"),
                   min_score: float = Form(0.0)):
        u = user_of(request)
        ds = dataset_or_404(did)
        rid_s, _, name = model.partition("|")
        runner = db.one("SELECT * FROM runners WHERE id = ?", int(rid_s or 0))
        if not runner:
            return back(f"/datasets/{did}", err="çalıştırıcı bulunamadı")
        if not can_use(u, runner):
            return back(f"/datasets/{did}", err="bu çalıştırıcıyı yalnız sahibi kullanabilir (sahibi paylaşmadı)")
        remote_id = int(name[1:]) if name.startswith("#") and name[1:].isdigit() else None
        remote = db.one("SELECT * FROM models WHERE id = ?", remote_id) if remote_id else None
        ad = ({"name": remote["name"], "version": remote["version"]} if remote else
              next((m for m in (db.runner_models(runner) if runner else []) if m["name"] == name), None))
        if not ad:
            return back(f"/datasets/{did}", err="çalıştırıcı ya da model bulunamadı")
        images = db.images(did)
        if scope == "missing":
            images = [i for i in images if ad["version"] not in db.predicted_versions(i["id"])]
        if not images:
            return back(f"/datasets/{did}", err="bu modelin ön etiketi olmayan görüntü yok (kapsamı 'tümü' seçin)")
        with db.conn() as c:
            jid = c.execute("INSERT INTO jobs (dataset_id, user_id, runner_id, model, model_version, remote_model, min_score, "
                            "created) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            (did, u["id"], runner["id"], ad["name"], ad["version"], remote_id,
                             max(0.0, min(1.0, min_score)), now())).lastrowid
            c.executemany("INSERT INTO job_items (job_id, image_id) VALUES (?, ?)", [(jid, i["id"]) for i in images])
        return back(f"/datasets/{did}", msg=f"iş {jid} kuyruğa alındı: {len(images)} görüntü, {runner['name']} / {name}")

    @app.post("/jobs/{jid}/cancel")
    def cancel_job(request: Request, jid: int):
        user_of(request)
        job = db.one("SELECT * FROM jobs WHERE id = ?", jid)
        if not job:
            raise HTTPException(404)
        if job["status"] in JOB_ACTIVE:
            db.run("UPDATE jobs SET status = 'cancelled', finished = ? WHERE id = ?", now(), jid)
        return back(f"/datasets/{job['dataset_id']}", msg=f"iş {jid} iptal edildi")

    @app.get("/api/datasets/{did}/jobs")
    def jobs_json(request: Request, did: int):
        user_of(request)
        jobs = db.jobs(did, 20)
        online = {r["id"]: r["online"] for r in runners_view()}
        for j in jobs:
            j["runner_online"] = online.get(j["runner_id"], False)
            j["errors"] = [dict(r) for r in db.all(
                "SELECT i.name, ji.error FROM job_items ji JOIN images i ON i.id = ji.image_id "
                "WHERE ji.job_id = ? AND ji.status = 'error' LIMIT 5", j["id"])]
        return {"jobs": jobs}

    # ------------------------------------------------------------------ karşılaştırma
    @app.post("/datasets/{did}/compare")
    def run_compare(request: Request, did: int, version: str = Form("")):
        u = user_of(request)
        ds = dataset_or_404(did)
        try:
            rid = svc.save_report(ds, u["id"], version or None)
        except (UserError, LSError) as e:
            return back(f"/datasets/{did}", err=str(e))
        return RedirectResponse(f"/reports/{rid}", status_code=303)

    def report_or_404(rid: int) -> dict:
        r = db.one("SELECT r.*, u.username AS user FROM reports r LEFT JOIN users u ON u.id = r.user_id "
                   "WHERE r.id = ?", rid)
        if not r:
            raise HTTPException(404, "rapor yok")
        r["summary"], r["rows"] = json.loads(r["summary"]), json.loads(r["rows"])
        return r

    @app.get("/reports/{rid}", response_class=HTMLResponse)
    def report_page(request: Request, rid: int):
        u = user_of(request)
        r = report_or_404(rid)
        ds = dataset_or_404(r["dataset_id"])
        tasks = {i["id"]: i["ls_task"] for i in db.images(ds["id"])}
        for row in r["rows"]:
            row["link"] = svc.task_link(ds["ls_project"], row["task"])
            row["has_image"] = row.get("image_id") in tasks
        return page(request, "report.html", u, r=r, ds=ds)

    @app.get("/reports/{rid}/report.json")
    def report_json(request: Request, rid: int):
        user_of(request)
        r = report_or_404(rid)
        return JSONResponse({"summary": r["summary"], "tasks": r["rows"], "model_version": r["model_version"]},
                            headers={"Content-Disposition": f'attachment; filename="report_{rid}.json"'})

    @app.get("/reports/{rid}/report.md")
    def report_md(request: Request, rid: int):
        user_of(request)
        r = report_or_404(rid)
        return Response(report_markdown(r["summary"], r["rows"]), media_type="text/markdown; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="report_{rid}.md"'})

    @app.get("/datasets/{did}/overlay/{iid}.png")
    def overlay(request: Request, did: int, iid: int, version: str = ""):
        user_of(request)
        ds = dataset_or_404(did)
        img = image_or_404(ds, iid)
        if not img["ls_task"]:
            raise HTTPException(404, "görev yok")
        return Response(svc.overlay(ds, img, version or None), media_type="image/png")

    @app.get("/datasets/{did}/export.zip")
    def export_zip(request: Request, did: int):
        user_of(request)
        ds = dataset_or_404(did)
        try:
            z = svc.export_zip(ds, request.query_params.getlist("f"))
        except (UserError, LSError) as e:
            return back(f"/datasets/{did}", err=str(e))
        return FileResponse(z, filename=z.name, background=BackgroundTask(lambda: _rmtree(z.parent)))

    # ------------------------------------------------------------------ çalıştırıcılar
    @app.get("/runners", response_class=HTMLResponse)
    def runners_page(request: Request):
        u = user_of(request)
        return page(request, "runners.html", u, runners=runners_view(), server_url=_server_url(request, s))

    @app.post("/runners", response_class=HTMLResponse)
    def create_runner(request: Request, name: str = Form("")):
        u = user_of(request)
        rid, token = auth.create_runner(db, u["id"], name or f"{u['username']} bilgisayarı")
        return page(request, "runners.html", u, runners=runners_view(), server_url=_server_url(request, s),
                    new_token=token, new_id=rid)

    @app.post("/runners/{rid}/delete")
    def delete_runner(request: Request, rid: int):
        u = user_of(request)
        r = db.one("SELECT * FROM runners WHERE id = ?", rid)
        if r and (u["admin"] or r["user_id"] == u["id"]):
            db.run("UPDATE jobs SET status = 'cancelled', finished = ? WHERE runner_id = ? AND status IN (?, ?)",
                   now(), rid, *JOB_ACTIVE)
            db.run("DELETE FROM runners WHERE id = ?", rid)
            return back("/runners", msg="çalıştırıcı silindi; anahtarı artık geçersiz")
        return back("/runners", err="çalıştırıcıyı yalnız sahibi ya da yönetici silebilir")

    @app.post("/runners/{rid}/share")
    def share_runner(request: Request, rid: int, shared: str = Form("")):
        u = user_of(request)
        r = db.one("SELECT * FROM runners WHERE id = ?", rid)
        if not r or r["user_id"] != u["id"]:
            return back("/runners", err="paylaşımı yalnız çalıştırıcının sahibi değiştirebilir")
        db.run("UPDATE runners SET shared = ? WHERE id = ?", int(bool(shared)), rid)
        return back("/runners", msg=f"{r['name']}: " + ("ekipteki herkes iş verebilir" if shared
                                                         else "artık yalnız siz iş verebilirsiniz"))

    @app.get("/download/runner.zip")
    def runner_zip(request: Request):
        user_of(request)
        return Response(_runner_zip(_server_url(request, s)), media_type="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="{RUNNER_DIR}.zip"'})

    @app.get("/baglan", response_class=HTMLResponse)
    def pair_page(request: Request, kod: str = ""):
        u = user_of(request)
        pairing = auth.pending_pairing(db, kod) if kod else None
        n = auth.normalize_code(kod)
        return page(request, "pair.html", u, code=f"{n[:4]}-{n[4:]}", pairing=pairing, expired=bool(kod) and not pairing)

    @app.post("/baglan")
    def pair_decide(request: Request, code: str = Form(...), action: str = Form(...), name: str = Form(""),
                    shared: str = Form("")):
        u = user_of(request)
        p = auth.pending_pairing(db, code)
        if not p:
            return back(f"/baglan?kod={quote(code)}", err="kod geçersiz ya da süresi dolmuş; çalıştırıcıyı yeniden başlatın")
        approve = action == "approve"
        name = name.strip() or p["info"].get("hostname") or f"{u['username']} bilgisayarı"
        auth.decide_pairing(db, p["id"], u["id"], name, approve, shared=bool(shared))
        if not approve:
            return back("/runners", msg="bağlantı isteği reddedildi")
        return back("/runners", msg=f"{name} hesabınıza bağlandı; birkaç saniye içinde çevrimiçi görünür")

    # ------------------------------------------------------------------ kullanıcılar
    @app.get("/users", response_class=HTMLResponse)
    def users_page(request: Request):
        u = admin_of(request)
        return page(request, "users.html", u, users=db.all("SELECT id, username, admin, created FROM users ORDER BY id"))

    @app.post("/users")
    def create_user(request: Request, username: str = Form(...), password: str = Form(...), admin: str = Form("")):
        admin_of(request)
        try:
            auth.create_user(db, username, password, bool(admin))
        except ValueError as e:
            return back("/users", err=str(e))
        return back("/users", msg=f"{username} eklendi")

    @app.post("/users/{uid}/delete")
    def delete_user(request: Request, uid: int):
        u = admin_of(request)
        if uid == u["id"]:
            return back("/users", err="kendinizi silemezsiniz")
        db.run("DELETE FROM users WHERE id = ?", uid)
        return back("/users", msg="kullanıcı silindi")

    @app.post("/users/{uid}/password")
    def reset_password(request: Request, uid: int, password: str = Form(...)):
        admin_of(request)
        if len(password) < 8:
            return back("/users", err="parola en az 8 karakter olmalı")
        db.run("UPDATE users SET pw = ? WHERE id = ?", auth.hash_password(password), uid)
        db.run("DELETE FROM sessions WHERE user_id = ?", uid)
        return back("/users", msg="parola değiştirildi")

    @app.get("/account", response_class=HTMLResponse)
    def account_page(request: Request):
        return page(request, "account.html", user_of(request))

    @app.post("/account/password")
    def change_password(request: Request, old: str = Form(...), new: str = Form(...)):
        u = user_of(request)
        row = db.one("SELECT pw FROM users WHERE id = ?", u["id"])
        if not auth.check_password(old, row["pw"]):
            return back("/account", err="mevcut parola yanlış")
        if len(new) < 8:
            return back("/account", err="yeni parola en az 8 karakter olmalı")
        db.run("UPDATE users SET pw = ? WHERE id = ?", auth.hash_password(new), u["id"])
        return back("/account", msg="parola değiştirildi")

    # ------------------------------------------------------------------ eşleştirme API'si (oturumsuz; kod + gizli anahtar)
    @app.post("/api/runner/pair")
    async def runner_pair_start(request: Request):
        body = await request.json()
        info = {k: str(v)[:100] for k, v in (body.get("info") or {}).items()
                if k in ("hostname", "platform", "python", "runner")}
        try:
            p = auth.start_pairing(db, info)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=429)
        return {**p, "url": f"{_server_url(request, s)}/baglan?kod={p['code']}", "interval": PAIR_POLL_SECONDS}

    @app.post("/api/runner/pair/poll")
    async def runner_pair_poll(request: Request):
        body = await request.json()
        return auth.claim_pairing(db, str(body.get("secret") or ""))

    # ------------------------------------------------------------------ çalıştırıcı API'si (Bearer anahtar)
    def runner_of(request: Request) -> dict:
        h = request.headers.get("authorization", "")
        r = auth.runner_by_token(db, h[7:].strip() if h.lower().startswith("bearer ") else None)
        if not r:
            raise HTTPException(401, "çalıştırıcı anahtarı geçersiz")
        db.run("UPDATE runners SET last_seen = ? WHERE id = ?", time.time(), r["id"])
        return r

    def active_job(runner_id: int, jid: int) -> dict:
        job = db.one("SELECT * FROM jobs WHERE id = ? AND runner_id = ?", jid, runner_id)
        if not job:
            raise HTTPException(404, "iş yok")
        return job

    def next_job(r: dict) -> dict | None:
        while True:
            job = db.one("SELECT * FROM jobs WHERE runner_id = ? AND status IN (?, ?) ORDER BY id LIMIT 1",
                         r["id"], *JOB_ACTIVE)
            if not job:
                return None
            items = db.all("SELECT i.id, i.name, i.width, i.height FROM job_items ji JOIN images i ON i.id = ji.image_id "
                           "WHERE ji.job_id = ? AND ji.status = 'pending' ORDER BY i.name", job["id"])
            if items:
                if job["status"] == "queued":
                    db.run("UPDATE jobs SET status = 'running', started = ? WHERE id = ?", now(), job["id"])
                return {"id": job["id"], "model": job["model"], "model_version": job["model_version"],
                        "remote_model": job["remote_model"],
                        "min_score": job["min_score"], "classes": s.classes, "items": items}
            _finish(job["id"], None)                        # her şey işlenmiş ama bitirilmemiş (ör. bağlantı koptu)

    def _finish(jid: int, error: str | None) -> None:
        done = db.one("SELECT COUNT(*) AS n FROM job_items WHERE job_id = ? AND status = 'done'", jid)["n"]
        status = "failed" if error or not done else "done"
        db.run("UPDATE jobs SET status = ?, error = ?, finished = ? WHERE id = ? AND status IN (?, ?)",
               status, error, now(), jid, *JOB_ACTIVE)

    @app.post("/api/runner/poll")
    async def runner_poll(request: Request):
        r = runner_of(request)
        body = await request.json()
        db.run("UPDATE runners SET models = ?, info = ? WHERE id = ?", json.dumps(body.get("models", [])),
               json.dumps(body.get("info", {})), r["id"])
        deadline = time.time() + min(float(body.get("wait", POLL_SECONDS)), POLL_SECONDS)
        while True:
            job = next_job(r)
            if job or time.time() >= deadline:
                return {"job": job}
            await asyncio.sleep(1)
            db.run("UPDATE runners SET last_seen = ? WHERE id = ?", time.time(), r["id"])

    @app.get("/api/runner/jobs/{jid}/images/{iid}")
    def runner_image(request: Request, jid: int, iid: int):
        r = runner_of(request)
        active_job(r["id"], jid)
        if not db.one("SELECT 1 FROM job_items WHERE job_id = ? AND image_id = ?", jid, iid):
            raise HTTPException(404, "görüntü bu işte yok")
        img = db.one("SELECT * FROM images WHERE id = ?", iid)
        return FileResponse(svc.path(img), filename=img["name"])

    @app.get("/api/runner/jobs/{jid}/model/{part}")
    def runner_model(request: Request, jid: int, part: str):
        r = runner_of(request)
        job = active_job(r["id"], jid)
        if job["status"] not in JOB_ACTIVE or not job["remote_model"] or part not in ("card", "weights"):
            raise HTTPException(404)
        m = db.one("SELECT * FROM models WHERE id = ?", job["remote_model"])
        if not m:
            raise HTTPException(404)
        return FileResponse(s.data_dir / m[part], filename=Path(m[part]).name)

    @app.post("/api/runner/jobs/{jid}/items/{iid}")
    async def runner_result(request: Request, jid: int, iid: int):
        r = runner_of(request)
        job = active_job(r["id"], jid)
        if job["status"] not in JOB_ACTIVE:
            return {"cancel": True}
        img = db.one("SELECT i.* FROM images i JOIN job_items ji ON ji.image_id = i.id "
                     "WHERE ji.job_id = ? AND i.id = ?", jid, iid)
        if not img:
            raise HTTPException(404, "görüntü bu işte yok")
        payload = await request.json()
        try:
            await asyncio.to_thread(svc.store_result, job, img, payload)
        except (UserError, LSError) as e:
            svc.store_result(job, img, {"error": f"sunucu: {e}"})
        job = db.one("SELECT status FROM jobs WHERE id = ?", jid)
        return {"cancel": job["status"] not in JOB_ACTIVE}

    @app.post("/api/runner/jobs/{jid}/finish")
    async def runner_finish(request: Request, jid: int):
        r = runner_of(request)
        active_job(r["id"], jid)
        body = await request.json()
        _finish(jid, body.get("error"))
        return {"ok": True}

    return app


def _ago(ts) -> str:
    if not ts:
        return "hiç"
    d = time.time() - float(ts)
    return "az önce" if d < 10 else f"{int(d)} sn önce" if d < 60 else f"{int(d // 60)} dk önce" if d < 3600 \
        else f"{int(d // 3600)} sa önce" if d < 86400 else f"{int(d // 86400)} gün önce"


def _server_url(request: Request, s: Settings) -> str:
    return s.public_url.rstrip("/") or str(request.base_url).rstrip("/")


def _rmtree(p: Path) -> None:
    import shutil
    shutil.rmtree(p, ignore_errors=True)


RUNNER_DIR = "eng-calistirici"
RUNNER_FILES = ("app/*.py", "runner/*.py", "models/*.json", "models/*.md", "labeling_config.xml",
                "requirements.txt", "requirements-runner.txt", "scripts/run_runner.sh", "scripts/run_runner.bat",
                "scripts/export_onnx.py")
# runner/paket/ altındaki başlatıcılar paketin köküne konur (çift tıklanan dosyalar)
RUNNER_LAUNCHERS = {"calistir.sh": ("ENG-Calistirici.command", "eng-calistirici.sh"),
                    "calistir.bat": ("ENG-Calistirici.bat",), "BENIOKU.txt": ("BENIOKU.txt",),
                    "modeller-BENIOKU.txt": ("modeller/BENIOKU.txt",)}


def _runner_zip(server_url: str) -> bytes:
    """Yerel çalıştırıcı paketi: uygulama kodu, model kartları, tek tıklık başlatıcılar ve sunucu adresi. Model
    ağırlıkları, config.yaml ve anahtar dahil değil; çalıştırıcı ilk açılışta tarayıcı onayıyla eşleşir."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for pat in RUNNER_FILES:
            for p in sorted(C.ROOT.glob(pat)):
                zf.write(p, f"{RUNNER_DIR}/{p.relative_to(C.ROOT).as_posix()}")
        for src, targets in RUNNER_LAUNCHERS.items():
            data = (C.ROOT / "runner" / "paket" / src).read_bytes().replace(b"\r\n", b"\n")
            for t in targets:
                info = zipfile.ZipInfo(f"{RUNNER_DIR}/{t}", time.localtime()[:6])
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (0o100755 if t.endswith((".command", ".sh")) else 0o100644) << 16
                crlf = t.endswith((".bat", ".txt"))             # Windows cmd ve Not Defteri için
                zf.writestr(info, data.replace(b"\n", b"\r\n") if crlf else data)
        zf.writestr(f"{RUNNER_DIR}/runner/server.json", json.dumps({"server": server_url}) + "\n")
    return buf.getvalue()
