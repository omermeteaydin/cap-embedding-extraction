# cap-embedding-extraction

CLIP ve Perception Encoder modelleriyle görüntü/metin embedding'i üreten
Novavision capsule paketi. Yapı, `cap-object-tracking` (ObjectTracking)
paketindeki gerçek konvansiyonlarla (PackageModel, PackageHelper,
Application().get_param(), Capsule/Executor akışı) hizalanmıştır.

- Girdi: tek bir görüntü **veya** serbest metin (`inputData`)
- Çıktı: float embedding vektörü (`outputEmbedding`) + meta bilgi (`outputMeta`)
- İki ayrı executor: `ClipEmbedding` ve `PerceptionEncoderEmbedding`
- Detaylı dokümantasyon: `DOCUMENTATION.md`

## GPU gerekiyor mu?

| Yöntem | GPU şart mı? |
|---|---|
| CLIP (ViT-B-16 / ViT-B-32 / RN50) | ❌ Hayır — CPU'da çalışır |
| Perception Encoder | ✅ Evet — CUDA zorunlu |

GPU'nuz yoksa **CLIP executor'ıyla başlayın**; Perception Encoder tarafı
şimdilik doğrulanmamış (bkz. aşağıdaki uyarı).

## Hızlı test (SDK olmadan, sadece open_clip ile)

```bash
pip install -r requirements.txt
python apps/quick_test.py --image resources/sample.jpg --version ViT-B-16
python apps/quick_test.py --text "a red buoy on water" --version ViT-B-16
```

## Gerçek platform akışı (SDK ile)

`apps/inference.py` — `ObjectTracking/apps/inference.py` ile aynı desende,
`PackageModel` üzerinden HTTP endpoint'ine istek gönderir.

## ⚠ Bilinmesi gerekenler

- `sdks.novavision` import'ları ve `PackageHelper.build_model()` /
  `Application().get_param()` / `Image.get_frame()` çağrıları,
  `ObjectTracking` (cap-object-tracking) paketindeki gerçek kullanımdan
  birebir örneklenmiştir.
- CLIP tarafı `open_clip_torch` ile tam çalışır durumdadır ve bu ortamda
  (ağırlık indirmeden, rastgele başlatılmış model ile) forward-pass
  mekaniği doğrulanmıştır.
- Perception Encoder tarafı Meta'nın `perception_models` paketine
  bağımlıdır ve bu ortamda test edilememiştir; `src/utils/utils.py`
  içindeki `_load_perception_encoder` fonksiyonu en iyi çaba
  (best-effort) yazılmıştır — üretime almadan önce mutlaka manuel
  doğrulama yapılmalı.
- Kendi `BoTSORTTracker/reid.py`'nizdeki appearance-extractor deseni
  (crop → batch → normalize → cache) referans alınmıştır.
