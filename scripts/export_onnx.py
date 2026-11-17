"""Bir model kartının (smp ya da TorchScript) ağını ONNX'e çevirir ve yanına ONNX kartını yazar.

    .venv/bin/python scripts/export_onnx.py models/lite_v8.json models/lite_v8_onnx/lite_v8.onnx

ONNX sürümü aynı ön işleme ve çözümlemeyi kullanır (kart kopyalanır, yalnız format ve weights değişir); onnxruntime ile
CPU'da genelde daha hızlıdır. Cellpose akış çözümlemesi yine torch ister.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from app.model import SegModel  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("card", type=Path)
    ap.add_argument("out", type=Path, help="yazılacak .onnx dosyası")
    ap.add_argument("--weights", help="model dosyası (boşsa karttaki)")
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args(argv)
    m = SegModel(a.card, a.weights, device="cpu")
    if m.format == "onnx":
        print("kart zaten ONNX", file=sys.stderr)
        return 1
    m.load()
    n_in = int(m.card.get("net", {}).get("in_channels", 3 if m.card.get("input", {}).get("channel") == "rgb" else 1))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(m._net, torch.zeros(1, n_in, 256, 256), str(a.out), input_names=["image"], output_names=["logits"],
                      opset_version=a.opset, dynamic_axes={"image": {2: "h", 3: "w"}, "logits": {2: "h", 3: "w"}},
                      dynamo=False)
    card = {k: v for k, v in m.card.items() if k != "net"}
    card.update(format="onnx", weights=a.out.name)
    a.out.with_suffix(".json").write_text(json.dumps(card, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{a.out} ({a.out.stat().st_size >> 20} MB) + {a.out.with_suffix('.json').name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
