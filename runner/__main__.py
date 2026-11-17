"""Yerel çalıştırıcı: modeli kullanıcının kendi bilgisayarında çalıştırır.

Web'den indirilen pakette ENG-Calistirici'ya çift tıklamak yeterlidir: sunucu adresi pakette gelir; ilk açılışta
tarayıcıda "Onayla" denince bu bilgisayar hesaba bağlanır ve anahtar kullanıcının ayar klasörüne kaydedilir (elle
anahtar kopyalanmaz). Modeller `modeller/` klasöründen bulunur: kartı `models/` içinde olan ağırlık (ör. lite_net.pt)
ya da yanında aynı adlı .json kartı olan .onnx/.pt dosyası. Gelişmiş kullanım:

    python -m runner [--server https://eng.ornek.com] [--token engr_...] \
        [--model models/lite_v8.json=/yol/lite_net.pt] [--model /yol/baska.onnx] [--model python:/yol/m.py:Sinif]
    python -m runner --cikis          # bu bilgisayarın kayıtlı bağlantısını sil

Sunucuya yalnız dışarı doğru HTTPS istekleri yapar (port açmak gerekmez): iş sorar (uzun sorgu), işteki görüntüleri
indirir, tahmin eder, Label Studio biçimindeki sonucu gönderir. Model dosyası bilgisayardan çıkmaz. Kesilirse yeniden
başlatın; yarım kalan iş kaldığı yerden sürer.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import socket
import sys
import tempfile
import time
import traceback
import webbrowser
from pathlib import Path

import httpx

from app import config as C, convert
from app.model import ModelError, load_model

VERSION = "2"
MODELS_DIR = C.ROOT / "modeller"                     # kullanıcının ağırlık dosyalarını bıraktığı klasör
CARDS_DIR = C.ROOT / "models"                        # paketle gelen model kartları
PACKAGE_SERVER = C.ROOT / "runner" / "server.json"   # web'den indirilen pakette sunucu adresi
WEIGHT_EXT = (".onnx", ".pt", ".pth")
REMOTE_FORMATS = ("onnx", "smp")                     # web'den gelen modellerde yalnız kod çalıştırmayan biçimler


class Unauthorized(RuntimeError):
    pass


class PairingError(RuntimeError):
    pass


def json_weights(card: Path) -> str:
    return Path(json.loads(card.read_text(encoding="utf-8"))["weights"]).name


def card_weights(card: Path) -> str | None:
    try:
        return json_weights(card)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def machine_info() -> dict:
    return {"hostname": socket.gethostname(), "platform": f"{platform.system()} {platform.machine()}",
            "python": platform.python_version(), "runner": VERSION}


# ------------------------------------------------------------------ yerel ayarlar (anahtar burada saklanır)
def config_path() -> Path:
    base = os.environ.get("APPDATA") if os.name == "nt" else os.environ.get("XDG_CONFIG_HOME")
    return Path(base or Path.home() / ".config") / "eng-label-runner" / "config.json"


def load_settings() -> dict:
    try:
        data = json.loads(config_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    data.setdefault("servers", {})
    data.setdefault("models", [])
    return data


def save_settings(data: dict) -> None:
    """Yalnız bu kullanıcının okuyabileceği dosyaya (0600) yazar."""
    p = config_path()
    p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = p.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, p)


def packaged_server() -> str | None:
    try:
        return json.loads(PACKAGE_SERVER.read_text(encoding="utf-8"))["server"]
    except (OSError, ValueError, KeyError, TypeError):
        return None


# ------------------------------------------------------------------ model bulma
def discover_models(folder: Path = MODELS_DIR, cards_dir: Path = CARDS_DIR) -> tuple[list[str], list[str]]:
    """`modeller/` klasörü -> (--model değerleri, uyarılar). Kart `modeller/` ya da `models/` içinde olabilir; ağırlık
    kartta yazan dosya adıyla `modeller/` içinde aranır."""
    if not folder.is_dir():
        return [], []
    specs, used = [], set()
    for card in sorted(folder.glob("*.json")) + sorted(cards_dir.glob("*.json")):
        w = folder / (card_weights(card) or "")
        if w.is_file() and w not in used:
            specs.append(f"{card}={w}")
            used.add(w)
    known = sorted({n for c in cards_dir.glob("*.json") if (n := card_weights(c))})
    notes = [f"{w.name} için model kartı yok: dosya adını kartta yazan ada çevirin ({', '.join(known)}) ya da "
             f"yanına {w.stem}.json kartı koyun" for w in sorted(folder.iterdir())
             if w.suffix.lower() in WEIGHT_EXT and w not in used]
    return specs, notes


def model_spec(raw: str, cards_dir: Path = CARDS_DIR, choose=None) -> str:
    """Pencereye sürüklenen yol -> --model değeri. Kart bulunamazsa `choose(kartlar)` ile sorulur."""
    raw = raw.strip()
    parts = shlex.split(raw, posix=os.name != "nt") if raw else []
    p = Path((parts[0] if parts else raw).strip('"\'')).expanduser()
    if not p.is_file():
        raise ModelError(f"dosya bulunamadı: {p}")
    if p.suffix.lower() == ".json" or p.with_suffix(".json").is_file():
        return str(p)
    cards = sorted(cards_dir.glob("*.json"))
    match = [c for c in cards if card_weights(c) == p.name]
    card = match[0] if match else choose(cards) if choose and cards else None
    if not card:
        raise ModelError(f"{p.name} için model kartı bulunamadı (models/README.md)")
    return f"{card}={p}"


def ask_model() -> str | None:
    """Model yoksa kullanıcıdan dosyayı pencereye sürüklemesini ister; boş geçilirse None."""
    print(f"\nBu bilgisayarda model bulunamadı. Model dosyanızı (ör. lite_net.pt) şu klasöre koyabilirsiniz:\n"
          f"  {MODELS_DIR}\nya da dosyayı şimdi bu pencereye sürükleyip Enter'a basın. Yalnız web'e yüklenen "
          f"modelleri kullanacaksanız boş bırakıp Enter'a basın.", flush=True)

    def choose(cards: list[Path]) -> Path | None:
        print("Bu dosya hangi model kartıyla açılsın?")
        for i, c in enumerate(cards, 1):
            print(f"  {i}) {c.stem}")
        a = input("Numara: ").strip()
        return cards[int(a) - 1] if a.isdigit() and 0 < int(a) <= len(cards) else None

    while True:
        raw = input("Model dosyası: ").strip()
        if not raw:
            return None
        try:
            return model_spec(raw, choose=choose)
        except (ModelError, ValueError, OSError) as e:
            print(f"olmadı: {e}")


def parse_models(specs: list[str], device: str, strict: bool | None = None) -> dict[str, object]:
    """--model değerleri -> ad: model. `kart.json=ağırlık` biçiminde ağırlık dosyası ayrıca verilebilir. Değer
    verilmezse config.yaml > model denenir. strict=False ise açılamayan model uyarıyla atlanır."""
    models = {}
    auto_default = not specs
    strict = not auto_default if strict is None else strict
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
            if strict:
                raise
            log(f"varsayılan yerel model yok ({e}); web'den yüklenen modeller yine çalıştırılabilir" if auto_default
                else f"model atlandı ({e})")
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
        self.info = machine_info()

    def ads(self) -> list[dict]:
        return [{"name": n, "version": m.version, "device": str(m.device), "kind": type(m).__name__}
                for n, m in self.models.items()]

    def poll(self, wait: float = 20) -> dict | None:
        r = self.http.post("/api/runner/poll", json={"models": self.ads(), "info": self.info, "wait": wait},
                           timeout=wait + 30)
        if r.status_code == 401:
            raise Unauthorized("anahtar geçersiz (401): çalıştırıcı web'den silinmiş olabilir")
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
                fmt = json.loads(card.read_text(encoding="utf-8")).get("format")
                if fmt not in REMOTE_FORMATS:                # TorchScript/pickle açılırken kod çalıştırabilir
                    raise ModelError(f"web'den gelen {fmt} modeli güvenlik nedeniyle çalıştırılmaz "
                                     f"(yalnız {', '.join(REMOTE_FORMATS)})")
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


# ------------------------------------------------------------------ eşleştirme (tarayıcıda onay)
def show_code(p: dict, open_browser: bool = True) -> None:
    print("\n" + "=" * 64 + "\n  Bu bilgisayarı ENG hesabınıza bağlayın:\n"
          f"  1) Tarayıcıda açın:  {p['url']}\n"
          f"  2) Sayfadaki kodun {p['code']} olduğunu kontrol edip Onayla'ya basın.\n"
          f"  Kod {int(p.get('expires_in', 600)) // 60} dakika geçerli; bu pencereyi kapatmayın.\n" + "=" * 64 + "\n",
          flush=True)
    if open_browser:
        try:
            webbrowser.open(p["url"])
        except Exception:
            pass


def pair(server: str, verify: bool = True, open_browser: bool = True, http=None, show=None,
         sleep=time.sleep) -> dict:
    """Kod alır, kullanıcı web'de onaylayana kadar bekler -> {token, runner_id, name}. Anahtar ekranda gösterilmez."""
    http = http or httpx.Client(base_url=server.rstrip("/"), timeout=30, verify=verify, follow_redirects=True)
    show = show or (lambda p: show_code(p, open_browser))
    while True:
        try:
            r = http.post("/api/runner/pair", json={"info": machine_info()})
            if r.status_code in (404, 405):
                raise PairingError("sunucu eşleştirmeyi desteklemiyor; web'den anahtar oluşturup --token ile verin")
            if r.status_code == 429:
                raise PairingError(r.json().get("error", "çok fazla istek"))
            r.raise_for_status()
            p = r.json()
            show(p)
            deadline = time.time() + float(p.get("expires_in", 600))
            while time.time() < deadline:
                sleep(float(p.get("interval", 2)))
                st = http.post("/api/runner/pair/poll", json={"secret": p["secret"]})
                st.raise_for_status()
                st = st.json()
                if st["status"] == "approved":
                    return st
                if st["status"] == "denied":
                    raise PairingError("bağlantı isteği web'de reddedildi")
                if st["status"] == "expired":
                    break
            log("kodun süresi doldu; yeni kod alınıyor")
        except (httpx.TransportError, httpx.HTTPStatusError) as e:
            log(f"sunucuya ulaşılamadı ({type(e).__name__}: {e}); 10 s sonra tekrar")
            sleep(10)


def local_models(a, settings: dict) -> dict:
    """--model verilmişse yalnız onlar; yoksa modeller/ klasörü + daha önce seçilen dosyalar (+ geliştirici
    kurulumunda config.yaml). Hiç yoksa ve pencere etkileşimliyse dosya sorulur."""
    if a.model:
        return parse_models(a.model, a.device)
    specs, notes = discover_models()
    for n in notes:
        log(n)
    specs += [s for s in settings["models"] if s not in specs]
    models = parse_models(specs, a.device, strict=False) if specs else {}
    if not models and not a.once and sys.stdin.isatty():
        spec = ask_model()
        if spec:
            models = parse_models([spec], a.device)
            settings = load_settings()
            settings["models"] = [*settings["models"], spec]
            save_settings(settings)
            log(f"model kaydedildi; sonraki açılışlarda sorulmayacak ({config_path()})")
    if not models and C.CONFIG.exists():
        models = parse_models([], a.device)
    if not models:
        log("yerel model yok; yalnız web'den yüklenen modeller çalıştırılabilir")
    return models


def read_token_file(path: Path, sleep=time.sleep) -> str:
    """Web uygulaması anahtarı yazana kadar bekler."""
    while True:
        try:
            token = path.read_text(encoding="utf-8").strip()
            if token:
                return token
        except OSError:
            pass
        log(f"anahtar dosyası bekleniyor: {path}")
        sleep(5)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m runner", description=__doc__.split("\n")[0])
    ap.add_argument("--server", default=os.environ.get("ENG_SERVER"),
                    help="web uygulamasının adresi (ENG_SERVER; indirilen pakette hazır gelir)")
    ap.add_argument("--token", default=os.environ.get("ENG_RUNNER_TOKEN"),
                    help="çalıştırıcı anahtarı (ENG_RUNNER_TOKEN); verilmezse tarayıcıda onayla eşleşir")
    ap.add_argument("--token-file", default=os.environ.get("ENG_RUNNER_TOKEN_FILE"),
                    help="anahtarı bu dosyadan oku; dosya değişirse yeniden okur (sunucu çalıştırıcısı konteyneri)")
    ap.add_argument("--model", action="append", default=[],
                    help="kart.json[=ağırlık], ağırlık.onnx (yanında .json kartı) ya da python:dosya.py:Sinif")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    ap.add_argument("--insecure", action="store_true", help="TLS sertifikasını doğrulama (yalnız deneme için)")
    ap.add_argument("--once", action="store_true", help="bekleyen işleri yapıp çık")
    ap.add_argument("--no-browser", action="store_true", help="eşleştirmede tarayıcıyı kendiliğinden açma")
    ap.add_argument("--cikis", action="store_true", help="bu bilgisayarın kayıtlı bağlantısını (anahtarını) sil")
    a = ap.parse_args(argv)
    server = (a.server or packaged_server() or "").rstrip("/")
    if not server:
        ap.error("--server gerekli (web'den indirilen pakette hazır gelir)")
    settings = load_settings()
    if a.cikis:
        settings["servers"].pop(server, None)
        save_settings(settings)
        log(f"{server} bağlantısı bu bilgisayardan silindi; web'deki çalıştırıcı kaydını da silebilirsiniz")
        return 0
    try:
        log(f"sunucu {server}")
        models = local_models(a, settings)
        while True:
            token = a.token or settings["servers"].get(server, {}).get("token")
            if a.token_file:
                token = read_token_file(Path(a.token_file))
            if not token:
                got = pair(server, verify=not a.insecure, open_browser=not a.no_browser)
                settings = load_settings()
                settings["servers"][server] = {"token": got["token"], "runner_id": got.get("runner_id"),
                                               "name": got.get("name")}
                save_settings(settings)
                token = got["token"]
                log(f"bağlandı: {got.get('name')}; bundan sonra onay sorulmaz")
            try:
                Runner(server, token, models, verify=not a.insecure).loop(a.once)
                return 0
            except Unauthorized as e:
                if a.token_file:                             # web anahtarı yeniledi: dosyadan yeniden oku
                    log(f"{e}; anahtar dosyası yeniden okunacak")
                    time.sleep(5)
                    continue
                if a.token:
                    print(f"hata: {e}", file=sys.stderr)
                    return 1
                log(f"{e}; yeniden eşleştiriliyor")
                settings["servers"].pop(server, None)
                save_settings(settings)
    except (ModelError, PairingError) as e:
        print(f"hata: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        log("durduruldu")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
