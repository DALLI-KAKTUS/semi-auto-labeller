"""Görüntü okuma: PNG / TIFF / JPG ve Zeiss CZI -> uint8 RGB.

CZI'ler Zen'in PNG dışa aktarmasıyla aynı görüntü eşlemesiyle (DisplaySetting: Low/High/Gamma, kanal rengi) okunur;
CAMEX'teki eğitim fotoğrafları bu dışa aktarmalardan yapıldığı için model aynı girdiyi görür (eşleme kuralı CAMEX
camex_desktop_app/server/pipeline.py'den alındı). 16 bit PNG/TIFF'ler 0.1-99.9 yüzdeliklerine göre 8 bite indirilir.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image

EXT = (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".czi")
BROWSER_EXT = (".png", ".jpg", ".jpeg")          # Label Studio'da (tarayıcıda) doğrudan gösterilebilenler


def _to_u8(a: np.ndarray) -> np.ndarray:
    if a.dtype == np.uint8:
        return a
    lo, hi = np.percentile(a, [0.1, 99.9])
    return np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1e-6) * 255, 0, 255).round().astype(np.uint8)


def read_pil(path: Path) -> np.ndarray:
    """PIL ile okunan görüntü -> uint8 HxWx3 (çok sayfalı TIFF'te ilk sayfa)."""
    with Image.open(path) as im:
        if im.mode in ("RGB", "L", "P", "RGBA", "LA", "CMYK", "YCbCr"):
            return np.asarray(im.convert("RGB"))
        a = np.asarray(im)                         # I;16, I, F: yüksek bit derinliği
    a = _to_u8(a[..., :3] if a.ndim == 3 else a)
    return np.repeat(a[..., None], 3, -1) if a.ndim == 2 else a


# ------------------------------------------------------------------ CZI
def czi_display_settings(meta: str) -> list[dict] | None:
    """Son <DisplaySetting> bloğundaki kanal eşlemeleri: [{low, high, gamma, bits, color}] (low/high bit aralığının
    kesri). Zen PNG dışa aktarırken bunu uygular."""
    i = meta.rfind("<DisplaySetting>")
    if i < 0:
        return None
    blk = meta[i:meta.find("</DisplaySetting>", i)]
    out = []
    for ch in re.findall(r"<Channel\b[^>]*>(.*?)</Channel>", blk, re.S):
        def val(k, d):
            m = re.search(rf"<{k}>([^<]+)</{k}>", ch)
            return float(m.group(1)) if m else d
        col = re.search(r"<Color>#([0-9A-Fa-f]{8}|[0-9A-Fa-f]{6})</Color>", ch)
        rgb = [int(col.group(1)[-6:][i:i + 2], 16) / 255 for i in (0, 2, 4)] if col else None
        out.append({"low": val("Low", 0.0), "high": val("High", 1.0), "gamma": val("Gamma", 1.0),
                    "bits": int(val("BitCountRange", 0)), "color": rgb})
    return out or None


def _apply_display(arr: np.ndarray, s: dict) -> np.ndarray:
    top = float(2 ** s["bits"] - 1) if s["bits"] else float(np.iinfo(arr.dtype).max if arr.dtype.kind in "ui" else 1.0)
    lo, hi = s["low"], s["high"] if s["high"] > s["low"] else s["low"] + 1e-6
    y = np.clip((arr.astype(np.float64) / top - lo) / (hi - lo), 0.0, 1.0) ** s["gamma"]
    return np.round(y * 255).astype(np.uint8)


def _legacy_u8(a: np.ndarray) -> np.ndarray:
    if a.dtype == np.uint8:
        return a
    hi = np.percentile(a, 99.9) or 1
    return np.clip(a.astype(np.float32) / hi * 255, 0, 255).astype(np.uint8)


def _czi_cyxs(arr: np.ndarray, axes: str) -> np.ndarray:
    """CZI dizisi -> (kanal, Y, X, örnek): diğer eksenlerin (sahne, zaman, z...) ilk dilimi."""
    a, ax = np.asarray(arr), list(axes)
    for i in reversed(range(len(ax))):
        if ax[i] not in "CYX0":
            a = np.take(a, 0, axis=i)
            del ax[i]
    if "C" not in ax:
        a, ax = a[None], ["C"] + ax
    if "0" not in ax:
        a, ax = a[..., None], ax + ["0"]
    return np.moveaxis(a, [ax.index(k) for k in "CYX0"], [0, 1, 2, 3])


def _czi_rgb(planes: np.ndarray, disp: list[dict] | None) -> tuple[np.ndarray, str]:
    C, _, _, S = planes.shape
    if C == 1 and S == 3:
        if disp:
            return _apply_display(planes[0], disp[0]), "czi_display_mapped"
        return _legacy_u8(planes[0]), "czi_no_display_setting"
    if C == 1:
        g = _apply_display(planes[0, ..., 0], disp[0]) if disp else _legacy_u8(planes[0, ..., 0])
        return np.stack([g] * 3, axis=-1), "czi_single_channel"
    comp = np.zeros(planes.shape[1:3] + (3,), np.float64)
    for c in range(C):
        ch = planes[c, ..., 0] if S == 1 else planes[c].mean(axis=-1).astype(planes.dtype)
        st = disp[c] if disp and c < len(disp) else {"low": 0.0, "high": 1.0, "gamma": 1.0, "bits": 0, "color": None}
        y = (_apply_display(ch, st) if disp else _legacy_u8(ch)).astype(np.float64) / 255
        comp += y[..., None] * np.asarray(st.get("color") or [1.0, 1.0, 1.0])
    return (np.clip(comp, 0, 1) * 255).round().astype(np.uint8), "czi_multichannel"


def read_czi(path: Path) -> tuple[np.ndarray, dict]:
    try:
        import czifile
    except ImportError as e:
        raise RuntimeError("CZI okumak için czifile gerekli: pip install czifile") from e
    with czifile.CziFile(str(path)) as czi:
        arr, meta, axes = czi.asarray(), czi.metadata(), czi.axes
    rgb, domain = _czi_rgb(_czi_cyxs(arr, axes), czi_display_settings(meta))
    info = {"input_domain": domain}
    m = re.search(r'<Distance Id="X">\s*<Value>([0-9.eE+-]+)</Value>', meta)
    if m:
        info["um_per_px"] = float(m.group(1)) * 1e6
    return np.ascontiguousarray(rgb), info


def read_image(path: str | Path) -> tuple[np.ndarray, dict]:
    """Görüntü -> (uint8 HxWx3, bilgi)."""
    path = Path(path)
    if path.suffix.lower() == ".czi":
        return read_czi(path)
    return read_pil(path), {"input_domain": path.suffix.lower().lstrip(".")}


def to_frame(rgb: np.ndarray, frame: tuple[int, int] | None) -> tuple[np.ndarray, int]:
    """Eğitim karesinin (G x Y) tam katı olan fotoğrafları blok ortalamasıyla o kareye indirir -> (rgb, kat)."""
    if not frame:
        return rgb, 1
    fw, fh = frame
    h, w = rgb.shape[:2]
    f = w // fw
    if f >= 2 and w == f * fw and h == f * fh:
        s = rgb.reshape(fh, f, fw, f, 3).astype(np.uint32).sum(axis=(1, 3))
        return ((s + f * f // 2) // (f * f)).astype(np.uint8), f
    return rgb, 1


def save_display(path: Path, out: Path) -> Path:
    """Tarayıcının gösteremediği görüntüyü (TIFF, CZI) PNG olarak yazar."""
    rgb, _ = read_image(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(out)
    return out
