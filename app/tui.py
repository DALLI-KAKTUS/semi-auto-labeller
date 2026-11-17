"""Textual arayüzü: Ayarlar, Veri, Tahmin, Label Studio, Durum, Dışa aktarma ve rapor.

Uzun işler (tahmin, yükleme, dışa aktarma, karşılaştırma) arka plan iş parçacığında çalışır; aynı anda tek iş.
    python -m app            # ya da: python -m app.tui
"""
from __future__ import annotations

import os
import traceback
import webbrowser
from pathlib import Path

from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import (Button, Checkbox, DataTable, Footer, Header, Input, Label, Markdown, ProgressBar,
                             RichLog, Select, Static, TabbedContent, TabPane)

from . import config as C, core, export
from .ls_client import LSError
from .model import ModelError, from_config

SETTINGS = [  # (input id, config yolu, etiket, parola mı)
    ("ls_url", ("label_studio", "url"), "Label Studio adresi", False),
    ("ls_key", ("label_studio", "api_key"), "API anahtarı (Personal Access Token)", True),
    ("ls_title", ("label_studio", "project_title"), "Yeni proje adı", False),
    ("ls_pid", ("label_studio", "project_id"), "Proje numarası (boş = ilk yüklemede oluştur)", False),
    ("m_card", ("model", "card"), "Model kartı (.json; uygulama klasörüne göre ya da mutlak)", False),
    ("m_weights", ("model", "weights"), "Model dosyası (boş = karttaki yol)", False),
    ("c_cell", ("classes", "cell"), "Hücre sınıf adı", False),
    ("c_branch", ("classes", "branch"), "Nörit sınıf adı", False),
    ("c_group", ("classes", "group"), "Grup sınıf adı", False),
    ("min_score", ("min_score",), "Güven eşiği (0-1, altındaki hücreler ön etikete konmaz)", False),
]


def _get(cfg: dict, path: tuple):
    for k in path:
        cfg = cfg[k]
    return cfg


def _set(cfg: dict, path: tuple, value) -> None:
    for k in path[:-1]:
        cfg = cfg[k]
    cfg[path[-1]] = value


class LabelApp(App):
    TITLE = "ENG label app"
    SUB_TITLE = "model ön etiketi -> Label Studio -> kontrol -> dışa aktarma"
    CSS = """
    TabbedContent { height: 1fr; }
    #log { height: 9; border-top: solid $primary; }
    .row { height: auto; margin: 0 0 1 0; }
    .row Button { margin-right: 1; }
    .field Label { margin-top: 1; color: $text-muted; }
    DataTable { height: 1fr; }
    ProgressBar { margin: 0 0 1 0; }
    #ls_info, #status_info, #data_info { margin: 0 0 1 0; }
    #report { height: 1fr; border: round $primary; }
    """
    BINDINGS = [("q", "quit", "Çıkış"), ("ctrl+s", "save_settings", "Ayarları kaydet")] + [
        (str(i), f"tab('{t}')", n) for i, (t, n) in enumerate(
            [("settings", "Ayarlar"), ("data", "Veri"), ("predict", "Tahmin"), ("ls", "Label Studio"),
             ("status", "Durum"), ("export", "Dışa aktar")], 1)]

    def __init__(self, config_path: Path = C.CONFIG):
        super().__init__()
        self.cfg = C.load(config_path)
        self.busy = False

    # -------------------------------------------------------------- yerleşim
    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(initial="data"):
            with TabPane("1 Ayarlar", id="settings"):
                with VerticalScroll(classes="field"):
                    for iid, path, label, pw in SETTINGS:
                        yield Label(label)
                        value = _get(self.cfg, path)
                        if iid == "ls_key" and self.cfg.get("_key_from_env"):
                            value = ""
                        yield Input(str(value if value is not None else ""), id=iid, password=pw,
                                    placeholder="LABEL_STUDIO_API_KEY ortam değişkeninden" if iid == "ls_key" else "")
                    yield Label("Cihaz")
                    yield Select([(d, d) for d in ("auto", "cpu", "cuda")], value=self.cfg["model"].get("device", "auto"),
                                 allow_blank=False, id="m_device")
                    with Horizontal(classes="row"):
                        yield Button("Kaydet", id="save", variant="primary")
                        yield Button("Bağlantıyı ve modeli test et", id="test")
            with TabPane("2 Veri", id="data"):
                with Horizontal(classes="row"):
                    yield Input(str(self.cfg["data"]["images"]), id="images", placeholder="görüntü klasörü")
                    yield Button("Tara", id="scan", variant="primary")
                yield Static("", id="data_info")
                yield DataTable(id="data_table", cursor_type="row")
            with TabPane("3 Tahmin", id="predict"):
                with Horizontal(classes="row"):
                    yield Button("Tahmin et", id="run_predict", variant="primary")
                    yield Checkbox("Yalnız tahmini olmayanlar", id="only_new", value=True)
                yield ProgressBar(id="predict_bar", show_eta=False)
                yield DataTable(id="predict_table", cursor_type="row")
            with TabPane("4 Label Studio", id="ls"):
                yield Static("", id="ls_info")
                with Horizontal(classes="row"):
                    yield Button("Projeye yükle", id="upload", variant="primary")
                    yield Button("Tarayıcıda aç", id="open")
                yield ProgressBar(id="upload_bar", show_eta=False)
            with TabPane("5 Durum", id="status"):
                with Horizontal(classes="row"):
                    yield Button("Yenile", id="refresh", variant="primary")
                    yield Button("Tarayıcıda aç", id="open2")
                yield Static("", id="status_info")
                yield DataTable(id="status_table", cursor_type="row")
            with TabPane("6 Dışa aktar / Rapor", id="export"):
                with Horizontal(classes="row"):
                    for f in export.FORMATS:
                        yield Checkbox(f, value=True, id=f"fmt_{f}")
                with Horizontal(classes="row"):
                    yield Button("Dışa aktar", id="run_export", variant="primary")
                    yield Button("Model ile insanı karşılaştır", id="run_eval")
                yield ProgressBar(id="export_bar", show_eta=False)
                with VerticalScroll(id="report"):
                    yield Markdown("Rapor henüz yok.", id="report_md")
        yield RichLog(id="log", markup=True, wrap=True)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#data_table", DataTable).add_columns("dosya", "boyut (KB)", "tahmin")
        self.query_one("#predict_table", DataTable).add_columns(
            "görüntü", "hücre", "eşik altı", "grup", "nörit px", "skor", "süre (s)", "hata")
        self.query_one("#status_table", DataTable).add_columns(
            "görev", "görüntü", "kontrol edildi", "model bölge", "insan bölge", "skor")
        self._update_ls_info()
        if self.cfg["data"]["images"]:
            self.scan()

    # -------------------------------------------------------------- yardımcılar
    def log_line(self, msg: str, style: str = "") -> None:
        self.query_one("#log", RichLog).write(f"[{style}]{msg}[/]" if style else msg)

    def action_tab(self, tab: str) -> None:
        # odak eski sekmede kalırsa Textual odağı orada bir sonraki öğeye taşır ve o sekmeyi geri açar
        self.set_focus(None)
        self.query_one(TabbedContent).active = tab

    def _progress(self, bar_id: str):
        bar = self.query_one(bar_id, ProgressBar)

        def cb(n: int, total: int, name: str):
            self.call_from_thread(bar.update, total=max(total, 1), progress=n)
        return cb

    def _log_cb(self, msg: str) -> None:
        self.call_from_thread(self.log_line, msg)

    def _start(self) -> bool:
        if self.busy:
            self.notify("Bir iş zaten çalışıyor", severity="warning")
            return False
        self.busy = True
        return True

    def _fail(self, e: Exception) -> None:
        known = isinstance(e, (LSError, ModelError, FileNotFoundError))
        msg = str(e) if known else f"{type(e).__name__}: {e}"
        self.call_from_thread(self.log_line, f"HATA: {msg}", "bold red")
        if not known:
            self.call_from_thread(self.log_line, traceback.format_exc(), "red")
        self.call_from_thread(self.notify, msg, severity="error", timeout=8)

    def _update_ls_info(self) -> None:
        ls = self.cfg["label_studio"]
        pid = ls.get("project_id")
        recs = core.predictions(self.cfg)
        txt = (f"Sunucu: {ls['url']}\nProje: {pid} ({ls['url']}/projects/{pid}/data)" if pid else
               f"Sunucu: {ls['url']}\nProje: yok, ilk yüklemede '{ls['project_title']}' adıyla oluşturulur")
        self.query_one("#ls_info", Static).update(f"{txt}\nYüklenecek tahmin: {len(recs)} görüntü")

    def _link(self) -> str | None:
        pid = self.cfg["label_studio"].get("project_id")
        return f"{self.cfg['label_studio']['url'].rstrip('/')}/projects/{pid}/data" if pid else None

    # -------------------------------------------------------------- ayarlar
    def action_save_settings(self) -> None:
        for iid, path, _, _ in SETTINGS:
            v = self.query_one(f"#{iid}", Input).value.strip()
            if iid == "ls_key":
                if not v:
                    self.cfg["_key_from_env"] = True
                    v = os.environ.get("LABEL_STUDIO_API_KEY", "")
                else:
                    self.cfg["_key_from_env"] = False
            elif iid == "ls_pid":
                v = int(v) if v.isdigit() else None
            elif iid == "min_score":
                try:
                    v = min(max(float(v), 0.0), 1.0)
                except ValueError:
                    self.notify("Güven eşiği sayı olmalı", severity="error")
                    return
            _set(self.cfg, path, v)
        self.cfg["model"]["device"] = self.query_one("#m_device", Select).value
        C.save(self.cfg)
        self._update_ls_info()
        self.log_line(f"ayarlar kaydedildi: {self.cfg['_path']}", "green")

    @on(Button.Pressed, "#save")
    def _save(self) -> None:
        self.action_save_settings()

    @on(Button.Pressed, "#test")
    def _test(self) -> None:
        self.action_save_settings()
        if self._start():
            self.test_connection()

    @work(thread=True, group="job")
    def test_connection(self) -> None:
        try:
            m = from_config(self.cfg)
            self._log_cb(f"model bulundu: {m.version} ({m.bundle})")
            self._log_cb("model yükleniyor (ilk seferde birkaç saniye)...")
            m.load()
            self._log_cb(f"model yüklendi, cihaz: {m.device}")
        except Exception as e:
            self._fail(e)
        try:
            email = core.client(self.cfg).check()
            self.call_from_thread(self.log_line, f"Label Studio bağlantısı tamam: {email}", "green")
        except Exception as e:
            self._fail(e)
        finally:
            self.busy = False

    # -------------------------------------------------------------- veri
    @on(Button.Pressed, "#scan")
    @on(Input.Submitted, "#images")
    def scan(self) -> None:
        folder = self.query_one("#images", Input).value.strip()
        table = self.query_one("#data_table", DataTable)
        table.clear()
        try:
            imgs = C.list_images(folder)
        except FileNotFoundError as e:
            self.query_one("#data_info", Static).update(str(e))
            return
        if folder != self.cfg["data"]["images"]:
            self.cfg["data"]["images"] = folder
            C.save(self.cfg)
        done = {Path(r["source"]).name for r in core.predictions(self.cfg).values()}
        for p in imgs:
            table.add_row(p.name, f"{p.stat().st_size / 1024:.0f}", "var" if p.name in done else "-")
        self.images = imgs
        self.query_one("#data_info", Static).update(
            f"{len(imgs)} görüntü, {sum(p.name in done for p in imgs)} tanesinin tahmini var  ({folder})")

    # -------------------------------------------------------------- tahmin
    @on(Button.Pressed, "#run_predict")
    def _predict(self) -> None:
        imgs = getattr(self, "images", [])
        if self.query_one("#only_new", Checkbox).value:
            done = {Path(r["source"]).name for r in core.predictions(self.cfg).values()}
            imgs = [p for p in imgs if p.name not in done]
        if not imgs:
            self.notify("Tahmin edilecek görüntü yok (Veri sekmesinde klasörü tarayın)", severity="warning")
            return
        if self._start():
            self.predict(imgs)

    @work(thread=True, group="job")
    def predict(self, imgs: list[Path]) -> None:
        table = self.query_one("#predict_table", DataTable)
        self.call_from_thread(table.clear)
        try:
            rows = core.run_predict(self.cfg, imgs, self._progress("#predict_bar"), self._log_cb)
            for r in rows:
                self.call_from_thread(table.add_row, r["image"], r.get("cells", ""), r.get("cells_below_threshold", ""),
                                      r.get("groups", ""), r.get("neurite_px", ""), r.get("score", ""),
                                      r.get("seconds", ""), r.get("error", ""))
            self.call_from_thread(self.notify, f"Tahmin bitti: {len(rows)} görüntü")
        except Exception as e:
            self._fail(e)
        finally:
            self.busy = False
            self.call_from_thread(self.scan)
            self.call_from_thread(self._update_ls_info)

    # -------------------------------------------------------------- Label Studio
    @on(Button.Pressed, "#upload")
    def _upload(self) -> None:
        if self._start():
            self.upload()

    @work(thread=True, group="job")
    def upload(self) -> None:
        try:
            pid = core.run_upload(self.cfg, self._progress("#upload_bar"), self._log_cb)
            self.call_from_thread(self._show_project, pid)
        except Exception as e:
            self._fail(e)
        finally:
            self.busy = False
            self.call_from_thread(self._update_ls_info)

    def _show_project(self, pid: int) -> None:
        self.query_one("#ls_pid", Input).value = str(pid)
        self.notify(f"Yüklendi: proje {pid}")

    @on(Button.Pressed, "#open, #open2")
    def _open(self) -> None:
        link = self._link()
        if link:
            webbrowser.open(link)
        else:
            self.notify("Henüz proje yok", severity="warning")

    # -------------------------------------------------------------- durum
    @on(Button.Pressed, "#refresh")
    def _refresh(self) -> None:
        if self._start():
            self.refresh_status()

    @on(TabbedContent.TabActivated)
    def _tab(self, event: TabbedContent.TabActivated) -> None:
        if event.pane.id == "status" and self.cfg["label_studio"].get("project_id") and not self.busy:
            self._refresh()

    @work(thread=True, group="job")
    def refresh_status(self) -> None:
        table = self.query_one("#status_table", DataTable)
        try:
            s = core.run_status(self.cfg)
            info = (f"Proje {s['project']}: toplam {s['total']} görev, kontrol edilen {s['labeled']}, "
                    f"bekleyen {s['pending']}\n{s['link']}")
            self.call_from_thread(self.query_one("#status_info", Static).update, info)
            self.call_from_thread(table.clear)
            for r in s["tasks"]:
                self.call_from_thread(table.add_row, r["task"], r["image"], "evet" if r["labeled"] else "hayır",
                                      r["regions_model"], "" if r["regions_human"] is None else r["regions_human"],
                                      "" if r["score"] is None else f"{r['score']:.3f}")
        except Exception as e:
            self._fail(e)
        finally:
            self.busy = False

    # -------------------------------------------------------------- dışa aktarma ve rapor
    @on(Button.Pressed, "#run_export")
    def _export(self) -> None:
        fmts = [f for f in export.FORMATS if self.query_one(f"#fmt_{f}", Checkbox).value]
        if not fmts:
            self.notify("En az bir format seçin", severity="warning")
            return
        if self._start():
            self.do_export(fmts)

    @work(thread=True, group="job")
    def do_export(self, fmts: list[str]) -> None:
        try:
            out = core.run_export(self.cfg, fmts, self._progress("#export_bar"), self._log_cb)
            self.call_from_thread(self.notify, f"Dışa aktarıldı: {out}")
        except Exception as e:
            self._fail(e)
        finally:
            self.busy = False

    @on(Button.Pressed, "#run_eval")
    def _eval(self) -> None:
        if self._start():
            self.do_evaluate()

    @work(thread=True, group="job")
    def do_evaluate(self) -> None:
        try:
            _, _, out = core.run_evaluate(self.cfg, self._progress("#export_bar"), self._log_cb)
            md = (out / "report.md").read_text(encoding="utf-8") + f"\n\nKayıt: `{out}`\n"
            self.call_from_thread(self.query_one("#report_md", Markdown).update, md)
        except Exception as e:
            self._fail(e)
        finally:
            self.busy = False


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(prog="eng-label-app")
    ap.add_argument("--config", default=str(C.CONFIG))
    LabelApp(Path(ap.parse_args().config)).run()


if __name__ == "__main__":
    main()
