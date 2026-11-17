"""Yerel çalıştırıcı: modeli kullanıcının kendi bilgisayarında çalıştırır.

    python -m runner --server https://eng.ornek.com --token engr_... \
        --model models/lite_v8.json=/yol/lite_net.pt [--model /yol/baska.onnx] [--model python:/yol/m.py:Sinif]

Sunucuya yalnız dışarı doğru HTTPS istekleri yapar (port açmak gerekmez): iş sorar (uzun sorgu), işteki görüntüleri
indirir, tahmin eder, Label Studio biçimindeki sonucu gönderir. Model dosyası bilgisayardan çıkmaz. --model verilmezse
config.yaml > model kullanılır. Kesilirse yeniden başlatın; yarım kalan iş kaldığı yerden sürer.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import tempfile
import time
import traceback
from pathlib import Path

import httpx

from app import config as C, convert
from app.model import ModelError, load_model

VERSION = "1"


def json_weights(card: Path) -> str:
    return Path(json.loads(card.read_text(encoding="utf-8"))["weights"]).name


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def parse_models(specs: list[str], device: str) -> dict[str, object]:
    """--model değerleri -> ad: model. `kart.json=ağırlık` biçiminde ağırlık dosyası ayrıca verilebilir."""
    models = {}
    auto_default = not specs
    if not specs:
        cfg = C.load()
        m = cfg["model"]
        specs = [f"{m['card']}={m['weights']}" if m.get("weights") else m["card"]] if m.get("card") else []
        device = device if device != "auto" else m.get("device", "auto")
    for spec in specs:
        card, _, weights = spec.partition("=") if not spec.startswith("python:") else (spec, "", "")
        try:
            model = load_model(card, weights or None, device, None, C.ROOT)
        except ModelError as e:
            if not auto_default:
                raise
            log(f"varsayılan yerel model yok ({e}); web'den yüklenen modeller yine çalıştırılabilir")
            continue
        name = getattr(model, "name", None) or model.version
        while name in models:
            name += "_2"
        models[name] = model
    return models


class Runner:
    def __init__(self, server: str, token: str, models: dict, verify: bool = True):
        self.server = server.rstrip("/")
        self.models = models
        self.http = httpx.Client(base_url=self.server, headers={"Authorization": f"Bearer {token}"}, timeout=120,
                                 verify=verify, follow_redirects=True)
        self.info = {"hostname": socket.gethostname(), "platform": f"{platform.system()} {platform.machine()}",
                     "python": platform.python_version(), "runner": VERSION}

    def ads(self) -> list[dict]:
        return [{"name": n, "version": m.version, "device": str(m.device), "kind": type(m).__name__}
                for n, m in self.models.items()]

    def poll(self, wait: float = 20) -> dict | None:
        r = self.http.post("/api/runner/poll", json={"models": self.ads(), "info": self.info, "wait": wait},
                           timeout=wait + 30)
        if r.status_code == 401:
            raise SystemExit("anahtar geçersiz (401): web arayüzünde Yerel çalıştırıcılar > Yeni anahtar oluştur")
        r.raise_for_status()
        return r.json().get("job")

    def run_job(self, job: dict) -> None:
        jid = job["id"]
        remote = job.get("remote_model")
        model = self.models.get(job["model"]) if not remote else None
        if remote:
            try:
                cache = Path.home() / ".cache" / "eng-label-app" / str(remote) / job["model_version"]
                cache.mkdir(parents=True, exist_ok=True)
                card = cache / "model.json"
                if not card.exists() or not (cache / json_weights(card)).is_file():
                    for part in ("card", "weights"):
                        with self.http.stream("GET", f"/api/runner/jobs/{jid}/model/{part}") as r:
                            r.raise_for_status()
                            name = "model.json" if part == "card" else Path(r.headers.get("content-disposition", "weights.pt").split('filename=')[-1].strip('"')).name
                            target = cache / name
                            with target.with_suffix(target.suffix + ".part").open("wb") as fh:
                                for chunk in r.iter_bytes(1 << 20):
                                    fh.write(chunk)
                            target.with_suffix(target.suffix + ".part").replace(target)
                model = load_model(str(card), device="auto")
                if model.version != job["model_version"]:
                    raise ModelError(f"model sürümü eşleşmiyor: {model.version} != {job['model_version']}")
                log(f"iş {jid}: model indirildi/önbellekten açıldı ({cache})")
            except (httpx.TransportError, httpx.HTTPStatusError):
                raise
            except Exception as e:
                log(f"iş {jid}: model indirilemedi: {e}")
                self.http.post(f"/api/runner/jobs/{jid}/finish", json={"error": f"model indirilemedi: {e}"})
                return
        if model is None:
            log(f"iş {jid}: bu çalıştırıcıda {job['model']} modeli yok")
            self.http.post(f"/api/runner/jobs/{jid}/finish", json={"error": f"çalıştırıcıda {job['model']} yok"})
            return
        log(f"iş {jid}: {len(job['items'])} görüntü, model {job['model']} ({model.version})")
        try:
            model.load()
        except Exception as e:
            log(f"iş {jid}: model yüklenemedi: {e}")
            self.http.post(f"/api/runner/jobs/{jid}/finish", json={"error": f"model yüklenemedi: {e}"})
            return
        with tempfile.TemporaryDirectory(prefix="eng_runner_") as tmp:
            for n, item in enumerate(job["items"], 1):
                t0 = time.time()
                try:
                    path = Path(tmp) / Path(item["name"]).name
                    with self.http.stream("GET", f"/api/runner/jobs/{jid}/images/{item['id']}") as r:
                        r.raise_for_status()
                        with open(path, "wb") as fh:
                            for chunk in r.iter_bytes(1 << 20):
                                fh.write(chunk)
                    pred = model.predict(path)
                    ls = convert.to_ls(pred, job["classes"], float(job.get("min_score") or 0.0))
                    summary = ls.pop("summary")
                    payload = {"prediction": ls, "summary": summary, "seconds": round(time.time() - t0, 2),
                               "meta": {k: v for k, v in pred.meta.items() if isinstance(v, (str, int, float, bool))}}
                    msg = (f"{summary['cells']} hücre, {summary['groups']} grup, {summary['neurite_px']} px nörit, "
                           f"{payload['seconds']} s")
                    path.unlink(missing_ok=True)
                except (httpx.TransportError, httpx.HTTPStatusError):
                    raise                                   # bağlantı sorunu: işi bırak, yeniden bağlanınca sürer
                except Exception as e:
                    payload = {"error": f"{type(e).__name__}: {e}"}
                    msg = f"HATA {payload['error']}"
                    traceback.print_exc()
                r = self.http.post(f"/api/runner/jobs/{jid}/items/{item['id']}", json=payload)
                r.raise_for_status()
                log(f"  [{n}/{len(job['items'])}] {item['name']}: {msg}")
                if r.json().get("cancel"):
                    log(f"iş {jid}: iptal edildi")
                    return
        self.http.post(f"/api/runner/jobs/{jid}/finish", json={})
        log(f"iş {jid}: bitti")

    def loop(self, once: bool = False) -> None:
        log(f"sunucu {self.server}; modeller: " + ", ".join(f"{a['name']} ({a['device']})" for a in self.ads()))
        delay = 5
        while True:
            try:
                job = self.poll(wait=2 if once else 20)
                delay = 5
                if job:
                    self.run_job(job)
                elif once:
                    return
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                log(f"sunucuya ulaşılamadı ({type(e).__name__}: {e}); {delay} s sonra tekrar")
                if once:
                    raise SystemExit(1)
                time.sleep(delay)
                delay = min(delay * 2, 60)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m runner", description=__doc__.split("\n")[0])
    ap.add_argument("--server", default=os.environ.get("ENG_SERVER"), help="web uygulamasının adresi (ENG_SERVER)")
    ap.add_argument("--token", default=os.environ.get("ENG_RUNNER_TOKEN"), help="çalıştırıcı anahtarı (ENG_RUNNER_TOKEN)")
    ap.add_argument("--model", action="append", default=[],
                    help="kart.json[=ağırlık], ağırlık.onnx (yanında .json kartı) ya da python:dosya.py:Sinif")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    ap.add_argument("--insecure", action="store_true", help="TLS sertifikasını doğrulama (yalnız deneme için)")
    ap.add_argument("--once", action="store_true", help="bekleyen işleri yapıp çık")
    a = ap.parse_args(argv)
    if not a.server or not a.token:
        ap.error("--server ve --token gerekli")
    try:
        models = parse_models(a.model, a.device)
    except ModelError as e:
        print(f"hata: {e}", file=sys.stderr)
        return 1
    try:
        Runner(a.server, a.token, models, verify=not a.insecure).loop(a.once)
    except KeyboardInterrupt:
        log("durduruldu")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
