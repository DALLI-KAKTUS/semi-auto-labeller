# Model kartları

Uygulama belirli bir CAMEX paketine bağlı değildir. Her modelin bir **ağırlık dosyası** ve modeli açıklayan bir JSON
**model kartı** vardır. Model kullanıcının bilgisayarında çalışır; sunucuda yalnız sonuçlar ve (web'den yüklenmişse)
model dosyasının bir kopyası tutulur. `models/lite_v8.json`, verilen `lite_net.pt` dosyasını çalıştıran örnek karttır;
CAMEX Python kaynak kodu gerekmez. CAMEX yolunu kendi bilgisayarınıza göre düzeltin veya `--model
models/lite_v8.json=/yol/lite_net.pt` kullanın.

## Kendi modeliniz (ONNX, smp, TorchScript)

Örnek: 4 kanallı (arka plan, hücre, nörit, grup) bir ONNX anlamsal segmentasyon ağı:

```json
{
  "name": "benim_modelim",
  "format": "onnx",
  "weights": "benim_modelim.onnx",
  "input": {
    "channel": "gray", "percentile": null, "scale": 255,
    "mean": 0, "std": 1, "pad": {"multiple": 32}
  },
  "output": {
    "class_channels": [0, 1, 2, 3], "activation": "softmax",
    "classes": {"cell": 1, "branch": 2, "group": 3}
  },
  "decode": {
    "cell": {"min_size": 20},
    "branch": {"min": 20}, "group": {"min": 100}
  }
}
```

- Girdi: `(1, C, H, W)` float32, çıktı: `(1, K, H, W)` logit; `class_channels` listedeki indisler modelin çıkış
  kanalları, `classes` ise bu listedeki **sıra numaralarıdır**. `input.channel`: `green` (varsayılan), `red`, `blue`,
  `gray` (RGB ortalaması), `max`, `rgb` (3 kanal). `percentile: [1,99.8]` yüzdelik normalizasyon; `null` olursa
  `scale` ile bölünür. `mean` ve `std` ardından uygulanır. `pad`: yansıtmalı dolgu, çıkarım sonrası kırpılır.
- `activation: softmax`: sınıflar birbirini dışlar; `thr` yoksa argmax. `sigmoid`: kanallar bağımsız, `thr` yoksa
  varsayılan 0.5. `cell` sınıfı bağlı bileşenlerden ayrı örnek hücrelere dönüştürülür. Nörit ve grup için
  `min` piksel altındaki 8-bağlı parçalar atılır; `bridge` (piksel) nörit uçlarını köprüler; `hyst: [alt, üst]`
  histerezis eşiğidir.
- Eğer ağ Cellpose akış başlığı da üretiyorsa `output.flow_channels: [Y, X]`, `cellprob_channel: N` verin.
  Hücreler bağlı bileşen yerine Cellpose akış çözümlemesiyle çıkarılır. `decode.cell`: `flow_threshold`,
  `cellprob_threshold`, `niter`, `min_size`, `circ_min`. Bu mod yerel çalıştırıcıda PyTorch/Cellpose ister.
- `format: smp` için `net: {"arch":"Unet", "encoder":"...", "in_channels":1, "classes":4,
  "state_dict_prefix":"model."}` ekleyin. Ağırlık `torch.save(state_dict)` (yalnız tensörler) biçiminde olmalı;
  `weights_only=True` ile açılır. `format: torchscript` için `torch.jit.save` dosyası gerekir. `net` gerekmez.
- `scripts/export_onnx.py kart.json çıktı.onnx [--weights model.pt]` smp/TorchScript ağını ONNX'e çevirir ve
  ONNX kartını yazar. Bu yalnız ağ katmanını dönüştürür; ön işleme ve Cellpose çözümlemesi Python'da kalır.

## Modeli ekleme yolları

1. **Web'den yükle:** Veri seti > Yeni model yükle: kart `.json` + ağırlık `.onnx`, `.pt` veya `.pth`. Model sunucuda
   saklanır; seçilen çalıştırıcı iş sırasında indirir ve **kendi bilgisayarında** çalıştırır. Güvenlik için yalnız
   `onnx` ve `smp` (yalnız tensör, `weights_only=True`) biçimleri kabul edilir; TorchScript açılırken kod
   çalıştırabileceğinden yalnız yerel kullanılır. Model dosyanızı sunucuda tutmak istemiyorsanız 2. yöntemi kullanın.
2. **Yalnız yerel:** Ağırlık dosyasını çalıştırıcı paketindeki `modeller/` klasörüne koyun. Dosya adı bir kartın
   `weights` adıyla aynıysa (ör. `lite_net.pt` → `models/lite_v8.json`) ya da yanında aynı adlı `.json` kartı varsa
   (`m.onnx` + `m.json`) çalıştırıcı açılırken bulur. Komut satırından `--model /yol/model.onnx` (yanında `model.json`)
   veya `--model /yol/kart.json=/yol/ağırlık.pt` de verilebilir; birden çok `--model` kullanılabilir. Model dosyası
   sunucuya hiç gönderilmez; çalıştırıcının ilan ettiği model veri seti sayfasındaki listede görünür.
3. **Özel Python modeli:** Yalnız yerel çalıştırıcıda `--model python:/yol/m.py:Sinif`. Sınıfın `version` özelliği,
   `predict(path)` yöntemi, isteğe bağlı `load()` yöntemi olmalı. `predict` bir `app.model.Prediction` veya
   `{"cell_labels": HxW uint16 (0=arka plan, 1..N=hücre), "branch": HxW bool,
   "group": HxW bool, "cell_scores": {1:0.9, ...}}` sözlüğü döndürmeli. Eksik maskeler boş kabul edilir.
   Python kodunu web sunucusuna yüklemeyin; güvenilmeyen Python/TorchScript modellerini çalıştırmayın.

Label Studio sınıf adları sabittir: `Cell`, `Neurite`, `Group` (`labeling_config.xml`). Karşılaştırma için hem
model hem insan aynı proje görevlerini kullanmalı. Model çıktısı görüntüyle aynı boyutta olmalıdır.
