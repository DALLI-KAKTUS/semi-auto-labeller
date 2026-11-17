"""Model katmanı: modelden bağımsız arayüz (predict(görüntü) -> etiketler) ve model kartıyla tanımlanan genel model.

Bir model, ağırlık dosyası + yanındaki JSON "model kartı"dır (örnek: models/lite_v8.json). Kart ağın nasıl kurulacağını
(smp state_dict, TorchScript ya da ONNX), girdinin nasıl hazırlanacağını ve çıktının nasıl etikete çevrileceğini söyler:

    {"name": "...", "weights": "ağırlık dosyası (karta göre ya da mutlak)",
     "format": "smp" | "torchscript" | "onnx",
     "net":    {"arch": "Unet", "encoder": "tu-convnext_tiny", "in_channels": 1, "classes": 7,
                "state_dict_prefix": "model."},                                  # yalnız smp
     "input":  {"channel": "green" | "red" | "blue" | "gray" | "max" | "rgb",
                "percentile": [1, 99.8] | null, "scale": 255,                    # yüzdelik normalizasyon ya da sabit bölen
                "mean": 0.449, "std": 0.226, "frame": [1920, 1080] | null,      # frame: bu karenin tam katları küçültülür
                "pad": {"multiple": 32, "sizes": {"1080x1920": [1280, 2048]}}},  # ortalanmış yansıtmalı dolgu
     "output": {"class_channels": [0, 1, 2, 3], "activation": "softmax" | "sigmoid",
                "classes": {"cell": 1, "branch": 2, "group": 3},                  # class_channels içindeki sıra
                "flow_channels": [4, 5], "cellprob_channel": 6,                   # Cellpose akışı (yoksa bağlı bileşen)
                "fp16_roundtrip": true},
     "decode": {"cell": {"flow_threshold": 0.4, "cellprob_threshold": -1.5, "niter": 200, "min_size": 0, "circ_min": 0},
                "branch": {"thr": null, "min": 100, "bridge": 10, "hyst": null},  # thr null = argmax
                "group": {"thr": 0.3, "min": 2000}}}

Hücre örnekleri akış kanalları varsa Cellpose akış dinamiğiyle, yoksa hücre sınıfının bağlı bileşenleriyle çıkarılır.
Ayrıca kendi Python sınıfınız da kullanılabilir (PluginModel). Maskeler her zaman okunan görüntünün boyutundadır.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from . import imageio


@dataclass
class Prediction:
    """Bir görüntünün model çıktısı; maskeler Label Studio'da gösterilecek görüntünün boyutundadır."""
    source: Path                     # kullanıcının seçtiği dosya
    display: Path                    # Label Studio'ya yüklenecek dosya (TIFF/CZI ise PNG kopyası)
    cell_labels: np.ndarray          # uint16, 0 = arka plan, 1..n = hücre
    cell_scores: dict[int, float]    # hücre no -> ortalama hücre olasılığı
    branch: np.ndarray               # bool, nörit
    group: np.ndarray                # bool, ayrılamayan hücre kümeleri
    model_version: str
    meta: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.cell_labels.shape


class Model(Protocol):
    version: str

    def load(self) -> None: ...

    def predict(self, path: Path) -> Prediction: ...


class ModelError(RuntimeError):
    pass


def _sha16(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def _display(path: Path, rgb: np.ndarray, upload_dir: Path | None) -> Path:
    """Tarayıcının gösteremediği girdiler (TIFF, CZI) için okunan görüntünün PNG kopyası."""
    if path.suffix.lower() in imageio.BROWSER_EXT or upload_dir is None:
        return path
    upload_dir.mkdir(parents=True, exist_ok=True)
    out = upload_dir / f"{path.stem}.png"
    Image.fromarray(rgb).save(out)
    return out


# ------------------------------------------------------------------ çözümleme adımları
def bridge_gaps(m: np.ndarray, max_gap: int) -> np.ndarray:
    """Farklı bağlı bileşenlerin iskelet uçları arasındaki <= max_gap px boşlukları 3 px kalınlığında doğruyla kapatır
    (her uç en fazla bir kez, en yakın çiftler önce)."""
    from scipy.spatial import cKDTree
    from skimage.draw import line
    from skimage.morphology import skeletonize
    sk = skeletonize(m)
    nb = ndi.convolve(sk.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - 1
    ep = np.argwhere(sk & (nb == 1))
    if len(ep) < 2:
        return m
    lab, _ = ndi.label(m, np.ones((3, 3), int))
    comp = lab[ep[:, 0], ep[:, 1]]
    pairs = sorted((np.hypot(*(ep[i] - ep[j])), i, j) for i, j in cKDTree(ep).query_pairs(max_gap) if comp[i] != comp[j])
    used, add = set(), np.zeros_like(m)
    for _, i, j in pairs:
        if i in used or j in used:
            continue
        rr, cc = line(*ep[i], *ep[j])
        add[rr, cc] = True
        used |= {i, j}
    return m | ndi.binary_dilation(add, iterations=1)


def binary_class(prob: np.ndarray, k: int, activation: str, thr: float | None = None, min_px: int = 0,
                 hyst: list | None = None, bridge: int = 0) -> np.ndarray:
    """k sınıfının ikili maskesi: argmax (thr yok, softmax), p > thr ya da histerezis (düşük, yüksek); sonra
    8-bağlı < min_px parçalar atılır, isteğe bağlı uç boşlukları köprülenir."""
    p = prob[k]
    if hyst:
        from skimage.filters import apply_hysteresis_threshold
        m = apply_hysteresis_threshold(p, hyst[0], hyst[1])
    elif thr is None:
        m = (prob.argmax(0) == k) if activation == "softmax" else p > 0.5
    else:
        m = p > thr
    if min_px:
        lab, n = ndi.label(m, np.ones((3, 3), int))
        if n:
            m = (np.bincount(lab.ravel()) >= min_px)[lab] & m
    return bridge_gaps(m, bridge) if bridge else m


def shape_filter(lab: np.ndarray, circ_min: float) -> np.ndarray:
    """Dairesellik 4*pi*A/P^2 < circ_min olan örnekleri atar (hücre sanılan nörit parçaları), 1..N yeniden numaralar."""
    if not circ_min:
        return lab
    from skimage.measure import regionprops
    from skimage.segmentation import relabel_sequential
    out = lab.copy()
    for r in regionprops(lab):
        per = r.perimeter or 1.0
        if min(1.0, 4 * np.pi * r.area / per ** 2) < circ_min:
            out[lab == r.label] = 0
    return relabel_sequential(out)[0].astype(lab.dtype)


def flow_cells(dP: np.ndarray, cellprob: np.ndarray, c: dict, device: str) -> np.ndarray:
    """Cellpose akış dinamiği -> hücre örnekleri (delikler doldurulur, küçükler atılır, yeniden numaralanır)."""
    import torch
    from cellpose import dynamics
    from cellpose.utils import fill_holes_and_remove_small_masks
    from skimage.segmentation import relabel_sequential
    lab = dynamics.compute_masks(dP, cellprob, niter=int(c.get("niter", 200)),
                                 cellprob_threshold=float(c.get("cellprob_threshold", 0.0)),
                                 flow_threshold=float(c.get("flow_threshold", 0.4)), min_size=-1,
                                 max_size_fraction=float(c.get("max_size_fraction", 0.4)), device=torch.device(device))
    lab = np.asarray(lab[0] if isinstance(lab, tuple) else lab)
    lab = fill_holes_and_remove_small_masks(lab.astype(np.int32), min_size=int(c.get("min_size", 0)))
    return shape_filter(relabel_sequential(lab)[0].astype(np.uint16), float(c.get("circ_min", 0.0)))


# ------------------------------------------------------------------ model kartıyla tanımlanan model
class SegModel:
    def __init__(self, card: str | Path, weights: str | Path | None = None, device: str = "auto",
                 upload_dir: Path | None = None):
        self.card_path = Path(card).expanduser()
        if not self.card_path.exists():
            raise ModelError(f"model kartı bulunamadı: {self.card_path}")
        try:
            self.card = json.loads(self.card_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ModelError(f"model kartı okunamadı ({self.card_path}): {e}") from e
        w = Path(weights or self.card.get("weights") or "").expanduser()
        self.weights = w if w.is_absolute() else (self.card_path.parent / w).resolve()
        if not self.weights.is_file():
            raise ModelError(f"model dosyası bulunamadı: {self.weights}")
        self.format = self.card.get("format") or ("onnx" if self.weights.suffix == ".onnx" else "smp")
        if self.format not in ("smp", "torchscript", "onnx"):
            raise ModelError(f"bilinmeyen model biçimi: {self.format}")
        self.name = self.card.get("name") or self.card_path.stem
        recipe = json.dumps({k: self.card.get(k) for k in ("format", "net", "input", "output", "decode")}, sort_keys=True)
        self.version = f"{self.name}-{_sha16(self.weights)[:8]}-{hashlib.sha256(recipe.encode()).hexdigest()[:6]}"
        self.bundle = self.weights
        self.upload_dir = upload_dir
        self._device = device
        self._net = None

    @property
    def device(self) -> str:
        if self.format == "onnx":
            return self._net.get_providers()[0] if self._net is not None else "onnxruntime"
        if self._device != "auto":
            return self._device
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

    def load(self) -> None:
        """Ağı belleğe alır (ilk tahmin beklemesin diye ayrı çağrılabilir)."""
        if self._net is not None:
            return
        try:
            if self.format == "onnx":
                import onnxruntime as ort
                prov = [p for p in ("CUDAExecutionProvider",) if p in ort.get_available_providers() and self._device != "cpu"]
                self._net = ort.InferenceSession(str(self.weights), providers=prov + ["CPUExecutionProvider"])
                return
            import torch
            if self.format == "torchscript":
                net = torch.jit.load(str(self.weights), map_location="cpu")
            else:
                import segmentation_models_pytorch as smp
                n = self.card["net"]
                net = smp.create_model(n["arch"], encoder_name=n["encoder"], encoder_weights=None,
                                       in_channels=int(n.get("in_channels", 1)), classes=int(n["classes"]))
                sd = torch.load(self.weights, map_location="cpu", weights_only=True)
                pre = n.get("state_dict_prefix", "")
                net.load_state_dict({k[len(pre):] if pre and k.startswith(pre) else k: v for k, v in sd.items()})
            self._net = net.to(self.device).eval()
        except ImportError as e:
            raise ModelError(f"model yığını eksik ({e}); requirements.txt kurulu mu?") from e
        except (RuntimeError, KeyError) as e:
            raise ModelError(f"model yüklenemedi ({self.weights.name}): {e}") from e

    def _forward(self, x: np.ndarray) -> np.ndarray:
        """(1, C, H, W) float32 -> (K, H, W) float32."""
        if self.format == "onnx":
            return self._net.run(None, {self._net.get_inputs()[0].name: x})[0][0].astype(np.float32)
        import torch
        with torch.no_grad():
            y = self._net(torch.from_numpy(x).to(self.device))
        y = y[0] if isinstance(y, (tuple, list)) else y
        return y[0].float().cpu().numpy()

    def _input(self, rgb: np.ndarray) -> np.ndarray:
        """uint8 HxWx3 -> (C, H, W) float32 [0, 1] (kanal seçimi + normalizasyon)."""
        i = self.card.get("input", {})
        ch = i.get("channel", "green")
        idx = {"red": 0, "green": 1, "blue": 2}
        g = rgb[..., idx[ch]] if ch in idx else rgb.max(-1) if ch == "max" else rgb if ch == "rgb" \
            else rgb.astype(np.float32).mean(-1)
        g = g.astype(np.float32)
        if i.get("percentile", [1.0, 99.8]):
            lo, hi = i.get("percentile", [1.0, 99.8])
            a, b = np.percentile(g, [lo, hi])
            g = np.clip((g - a) / max(b - a, 1e-6), 0, 1)
        else:
            g = g / float(i.get("scale", 255))
        g = np.clip(g, 0, 1).astype(np.float32)
        return g.transpose(2, 0, 1) if g.ndim == 3 else g[None]

    def _pads(self, h: int, w: int) -> tuple:
        p = self.card.get("input", {}).get("pad", {"multiple": 32})
        H, W = p.get("sizes", {}).get(f"{h}x{w}", (h + (-h) % int(p.get("multiple", 1)),
                                                      w + (-w) % int(p.get("multiple", 1))))
        ph, pw = H - h, W - w
        return (ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)

    def raw(self, rgb: np.ndarray) -> np.ndarray:
        """uint8 HxWx3 -> (H, W, K) ağ çıktısı."""
        i, o = self.card.get("input", {}), self.card.get("output", {})
        x = self._input(rgb)
        pads = self._pads(*x.shape[1:])
        x = np.stack([np.pad(c, pads, mode="reflect") for c in x])
        x = (x - np.float32(i.get("mean", 0.0))) / np.float32(i.get("std", 1.0)) \
            if np.isscalar(i.get("mean", 0.0)) else \
            (x - np.asarray(i["mean"], np.float32)[:, None, None]) / np.asarray(i["std"], np.float32)[:, None, None]
        y = self._forward(np.ascontiguousarray(x[None], dtype=np.float32)).transpose(1, 2, 0)
        (t, b), (l, r) = pads
        y = np.ascontiguousarray(y[t:y.shape[0] - b, l:y.shape[1] - r], dtype=np.float32)
        return y.astype(np.float16).astype(np.float32) if o.get("fp16_roundtrip") else y

    def decode(self, y: np.ndarray) -> dict:
        o, d = self.card.get("output", {}), self.card.get("decode", {})
        act = o.get("activation", "softmax")
        cls = o.get("classes", {"cell": 1, "branch": 2, "group": 3})
        z = y[..., o.get("class_channels", list(range(y.shape[-1])))]
        if act == "softmax":
            z = z - z.max(-1, keepdims=True)
            e = np.exp(z)
            prob = (e / e.sum(-1, keepdims=True)).transpose(2, 0, 1)
        else:
            prob = (1 / (1 + np.exp(-z))).transpose(2, 0, 1)
        h, w = y.shape[:2]
        out = {"cell_labels": np.zeros((h, w), np.uint16), "cell_scores": {},
               "branch": np.zeros((h, w), bool), "group": np.zeros((h, w), bool)}
        c = d.get("cell", {})
        if o.get("flow_channels"):
            dP = np.ascontiguousarray(y[..., o["flow_channels"]].transpose(2, 0, 1))
            cp = np.ascontiguousarray(y[..., o["cellprob_channel"]])
            cells = flow_cells(dP, cp, c, "cpu" if self.format == "onnx" else self.device)
            p_cell = (1.0 / (1.0 + np.exp(-cp.astype(np.float64)))).astype(np.float32)
        elif "cell" in cls:
            m = binary_class(prob, cls["cell"], act, c.get("thr"), 0, c.get("hyst"))
            lab, _ = ndi.label(m)
            if c.get("min_size"):
                lab = np.where((np.bincount(lab.ravel()) >= int(c["min_size"]))[lab], lab, 0)
            from skimage.segmentation import relabel_sequential
            cells = shape_filter(relabel_sequential(lab)[0].astype(np.uint16), float(c.get("circ_min", 0.0)))
            p_cell = prob[cls["cell"]]
        else:
            cells, p_cell = out["cell_labels"], None
        ids = [int(i) for i in np.unique(cells) if i]
        out["cell_labels"] = cells.astype(np.uint16)
        out["cell_scores"] = {i: float(m) for i, m in zip(ids, ndi.mean(p_cell, cells, ids))} if ids else {}
        for key in ("branch", "group"):
            if key in cls:
                s = d.get(key, {})
                out[key] = binary_class(prob, cls[key], act, s.get("thr"), int(s.get("min", 0)), s.get("hyst"),
                                        int(s.get("bridge", 0)))
        return out

    def predict(self, path: Path) -> Prediction:
        self.load()
        path = Path(path)
        rgb, info = imageio.read_image(path)
        small, f = imageio.to_frame(rgb, self.card.get("input", {}).get("frame"))
        out = self.decode(self.raw(small))
        cells, branch, group = out["cell_labels"], out["branch"], out["group"]
        if f > 1:                                    # model küçültülmüş karede çalıştı; maskeleri özgün boyuta büyüt
            up = lambda a: np.repeat(np.repeat(a, f, 0), f, 1)
            cells, branch, group = up(cells), up(branch), up(group)
        return Prediction(path, _display(path, rgb, self.upload_dir), cells, out["cell_scores"], branch, group,
                          self.version, {"device": self.device, "frame_factor": f, **info})


# ------------------------------------------------------------------ kullanıcının Python sınıfı
class PluginModel:
    """`python:/yol/model.py:Sinif`. Sınıf argümansız (ya da device=...) kurulur; `version` özniteliği ve
    `predict(path)` yöntemi olmalı. predict bir Prediction ya da {"cell_labels": uint16 HxW, "branch": bool HxW,
    "group": bool HxW, "cell_scores": {no: skor}} sözlüğü döndürür (eksik anahtarlar boş sayılır; maskeler
    imageio.read_image'in okuduğu görüntüyle aynı boyutta olmalı)."""

    def __init__(self, file: str | Path, cls: str, device: str = "auto", upload_dir: Path | None = None):
        self.file = Path(file).expanduser()
        if not self.file.exists():
            raise ModelError(f"eklenti dosyası bulunamadı: {self.file}")
        spec = importlib.util.spec_from_file_location(f"eng_plugin_{self.file.stem}", self.file)
        mod = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(self.file.parent))
        spec.loader.exec_module(mod)
        if not hasattr(mod, cls):
            raise ModelError(f"{self.file} içinde {cls} yok")
        try:
            self.impl = getattr(mod, cls)(device=device)
        except TypeError:
            self.impl = getattr(mod, cls)()
        self.name = cls
        self.version = str(getattr(self.impl, "version", f"{self.file.stem}-{cls}"))
        self.bundle = self.file
        self.device = device
        self.upload_dir = upload_dir

    def load(self) -> None:
        if hasattr(self.impl, "load"):
            self.impl.load()

    def predict(self, path: Path) -> Prediction:
        path = Path(path)
        out = self.impl.predict(path)
        if isinstance(out, Prediction):
            return out
        shape = next(np.asarray(out[k]).shape for k in ("cell_labels", "branch", "group") if out.get(k) is not None)
        cells = np.asarray(out["cell_labels"], np.uint16) if out.get("cell_labels") is not None else np.zeros(shape, np.uint16)
        empty = np.zeros(shape, bool)
        ids = [int(i) for i in np.unique(cells) if i]
        scores = {int(k): float(v) for k, v in (out.get("cell_scores") or {i: 1.0 for i in ids}).items()}
        display = path
        if path.suffix.lower() not in imageio.BROWSER_EXT and self.upload_dir is not None:
            display = _display(path, imageio.read_image(path)[0], self.upload_dir)
        return Prediction(path, display, cells, scores,
                          np.asarray(out["branch"], bool) if out.get("branch") is not None else empty,
                          np.asarray(out["group"], bool) if out.get("group") is not None else empty,
                          self.version, {"device": self.device})


def load_model(spec: str, weights: str | None = None, device: str = "auto", upload_dir: Path | None = None,
               base: Path | None = None):
    """Model tanımı -> model nesnesi.

        <kart.json>                model kartı (ağırlık dosyası kartta ya da `weights` ile)
        <ağırlık.onnx|.pt>         yanında aynı adlı .json kartı olan ağırlık dosyası
        python:<dosya.py>:<Sınıf>  kendi Python sınıfınız
    Göreli yollar `base` klasörüne göredir.
    """
    if spec.startswith("python:"):
        file, _, cls = spec[len("python:"):].rpartition(":")
        if not file or not cls:
            raise ModelError("python modeli `python:/yol/dosya.py:SinifAdi` biçiminde verilmeli")
        return PluginModel(_abs(file, base), cls, device, upload_dir)
    p = _abs(spec, base)
    if p.suffix.lower() != ".json":
        card = p.with_suffix(".json")
        if not card.exists():
            raise ModelError(f"{p.name} için model kartı yok: {card} (models/README örneklerine bakın)")
        return SegModel(card, weights or p, device, upload_dir)
    return SegModel(p, _abs(weights, base) if weights else None, device, upload_dir)


def _abs(path: str, base: Path | None) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() or base is None else base / p


def from_config(cfg: dict, upload_dir: Path | None = None):
    from .config import ROOT
    m = cfg["model"]
    if not m.get("card"):
        raise ModelError("model seçilmedi: config.yaml > model.card (model kartı) gerekli")
    return load_model(m["card"], m.get("weights") or None, m.get("device", "auto"), upload_dir, ROOT)


def save_masks(pred: Prediction, path: Path) -> None:
    np.savez_compressed(path, cell_labels=pred.cell_labels, branch=pred.branch, group=pred.group)


def load_masks(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}
