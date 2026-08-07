# EmbeddingExtraction - DOCUMENTATION

## 1. Genel Bakış

### Paketin amacı ve ne yaptığı

EmbeddingExtraction paketi, görüntülerden ve/veya serbest metinden **CLIP** ya da **Perception Encoder** modelleriyle semantik embedding vektörü üreten bir capsule uygulamasıdır. `cap-object-tracking` (ObjectTracking) paketinin gerçek konvansiyonlarıyla (PackageModel/PackageHelper/Application().get_param()/Capsule-Executor akışı) hizalanmıştır. Bu paket:

- Tek bir görüntü veya metin kabul eder
- İki ayrı executor sunar: `ClipEmbedding` ve `PerceptionEncoderEmbedding`
- Çıktıyla birlikte hangi model/versiyonla üretildiğine dair meta bilgi döner
- Opsiyonel L2-normalizasyon uygular

### Temel özellikler

- ✅ Görüntüden ve metinden semantik embedding çıkarımı
- ✅ CLIP versiyonları: ViT-B-16, ViT-B-32, RN50 (CPU'da çalışır, GPU şart değil)
- ✅ Perception Encoder versiyonları: PE-Core-B16-224, PE-Core-L14-336 (⚠ GPU/CUDA zorunlu)
- ✅ L2-normalizasyon (açılıp kapatılabilir config)
- ✅ `ObjectTracking`'deki "Advance" toggle deseniyle birebir aynı, `restart=True` taşıyan iç içe config yapısı (`ConfigClipAdvance` → True/False varyantları, bkz. §5.1 ve §9.3)
- ✅ Bootstrap'te model cache'leme (her `run()` çağrısında yeniden yüklenmez)
- ✅ `PackageHelper.build_model()` ile gerçek Novavision response mekanizması

### Desteklenen sınıflar / modeller / tipler

| ID | İsim | Açıklama |
|----|------|---------|
| 1  | `ClipEmbedding` | CLIP tabanlı embedding executor'ü — `src/executors/ClipEmbedding.py` |
| 2  | `PerceptionEncoderEmbedding` | Perception Encoder tabanlı embedding executor'ü (⚠ GPU zorunlu, doğrulanmamış) |
| 3  | `PackageModel` | Paket genel yapı tanımı (configs, executor) |
| 4  | `InputData` | Pydantic input modeli — Image veya serbest metin (Union) |
| 5  | `EmbeddingModelLoader` | Model yükleme/cache/inference sınıfı (`utils/utils.py`) |
| 6  | `ConfigClipVersion` / `ConfigPerceptionEncoderVersion` | Model varyantı seçimi |
| 7  | `ConfigClipNormalize` / `ConfigPerceptionEncoderNormalize` | L2-normalizasyon aç/kapa |
| 8  | `OutputEmbedding` | Float embedding vektörü çıktısı |
| 9  | `OutputMeta` | Model ailesi/versiyonu/boyut bilgisi |

---

## 2. Mimari ve Teknolojiler

### Teknoloji Stack'i
- Framework: Python 3.9+
- Model kütüphaneleri: `open_clip_torch` (CLIP, doğrulandı), Meta `perception_models` (Perception Encoder, ⚠ doğrulanmadı)
- Görüntü işleme: Pillow, NumPy, OpenCV (`sdks.novavision` üzerinden)
- Derin öğrenme: PyTorch (CPU veya CUDA)
- API: `sdks.novavision` (Capsule, Executor, PackageHelper, Application, Image) — `ObjectTracking` paketiyle aynı SDK kullanımı

### Proje yapısı (tree formatında)

```
cap-embedding-extraction/
├── LICENSE                              # MIT (DigiNova)
├── README.md
├── DOCUMENTATION.md                     # (Bu dosya)
├── setup.py                             # novavision.cap.embedding-extraction paket dizini
├── requirements.txt
├── .gitignore
├── __init__.py
├── apps/
│   ├── inference.py                     # Gerçek platform HTTP client örneği (SDK ile)
│   └── quick_test.py                    # SDK'sız hızlı yerel test (open_clip doğrudan)
├── resources/                           # Örnek input görseller
├── src/
│   ├── __init__.py
│   ├── executors/
│   │   ├── ClipEmbedding.py             # CLIP executor
│   │   └── PerceptionEncoderEmbedding.py # Perception Encoder executor
│   ├── models/
│   │   └── PackageModel.py              # Pydantic modeller
│   └── utils/
│       ├── response.py                  # PackageHelper ile response inşası
│       └── utils.py                     # EmbeddingModelLoader, config->cfg dönüşümü
```

Açıklamalar:
- `ClipEmbedding.py` / `PerceptionEncoderEmbedding.py` — her biri kendi `Capsule` alt sınıfı; `bootstrap()` modeli bir kez yükler, `run()` her istek için embedding üretir. `ObjectTracking/src/executors/BoTSortTracking.py` ile aynı iskelet.
- `PackageModel.py` — Pydantic modeller: her executor için ayrı Request/Response/Executor üçlüsü, `ObjectTracking`'deki gibi.
- `utils/utils.py` — `EmbeddingModelLoader` sınıfı + `Application().get_param()` ile config okuyan `_build_clip_cfg`/`_build_perception_encoder_cfg` fonksiyonları (`_build_bot_sort_cfg` ile aynı desen).
- `utils/response.py` — `PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs).build_model(context)` ile gerçek response nesnesini kurar.

---

## 3. Executor'lar ve Çalışma Modları

### `ClipEmbedding` (Tam path: `src/executors/ClipEmbedding.py`)

- Amaç: Görüntü veya metinden CLIP ile semantik embedding üretmek.
- Kullanım senaryosu:
  - ✅ Zero-shot görsel/metinsel benzerlik arama
  - ✅ Detection-crop üzerinde kaba görünüm imzası çıkarma
  - ✅ GPU'suz makinelerde (ör. yerel geliştirme ortamı) çalıştırılabilir tek embedding yöntemi
- İşleyiş (numaralı adımlar):
  1. `bootstrap(config)` → `Application().get_param()` ile `ConfigClipAdvance` okunur; `True` ise `ClipVersion`/`ClipNormalize`, `False` ise varsayılanlar kullanılır; `EmbeddingModelLoader` oluşturulur (normalize bootstrap-time'da loader'a sabitlenir)
  2. `__init__` → `self.request.get_param("inputData")` ile SADECE input alınır (config DEĞİL — bkz. §9)
  3. `run()`:
     1. Girdi görüntü ise `Image.get_frame(img=..., redis_db=self.redis_db)` ile frame alınır
     2. Girdi metin ise doğrudan string olarak işlenir
     3. `loader.embed_image()` / `loader.embed_text()` çağrılır (normalize parametresi verilmez, loader'ın bootstrap-time normalize'ı kullanılır)
     4. `build_clip_response()` ile `PackageHelper` üzerinden gerçek Response inşa edilir
- Temel metodlar: `__init__`, `bootstrap(config)` (staticmethod), `run(self)`
- Dosya sonu: `if "__main__" == __name__: Executor(sys.argv[1]).run()`

### `PerceptionEncoderEmbedding` (Tam path: `src/executors/PerceptionEncoderEmbedding.py`)

- Amaç: Meta Perception Encoder ile CLIP'e alternatif embedding üretmek.
- ⚠ GPU/CUDA zorunlu; bu ortamda `perception_models` paketi kurulu olmadığından **uçtan uca doğrulanamadı**.
- İşleyiş: `ClipEmbedding` ile birebir aynı akış, sadece `load_perception_encoder_loader()` çağrılır.

---

## 4. Girdi (Input) Parametreleri

### 4.1 `InputData`
```python
class InputData(Input):
    name: Literal["inputData"] = "inputData"
    value: Union[Image, str]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get('value')
        if isinstance(value, Image):
            return "object"
        return "string"

    class Config:
        title = "Data"
```
- Tanım: Tek bir görüntü veya serbest metin (Union tipi)
- Kullanıldığı executor'lar: ClipEmbedding ✅, PerceptionEncoderEmbedding ✅

---

## 5. Konfigürasyon (Config) Parametreleri

### 5.1 `ConfigClipAdvance` / `ConfigPerceptionEncoderAdvance`
```python
class ConfigClipAdvanceTrue(Config):
    name: Literal["True"] = "True"
    value: Literal["True"] = "True"
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"
    configClipVersion: ConfigClipVersion
    configClipNormalize: ConfigClipNormalize

class ConfigClipAdvanceFalse(Config):
    name: Literal["False"] = "False"
    value: Literal["False"] = "False"
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

class ConfigClipAdvance(Config):
    name: Literal["ConfigClipAdvance"] = "ConfigClipAdvance"
    value: Union[ConfigClipAdvanceTrue, ConfigClipAdvanceFalse]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"
    restart: Literal[True] = True
```
- Tanım: `ObjectTracking`'deki `ConfigBoTSortAdvance` ile birebir aynı desen. `False` ise varsayılan değerler (ViT-B-16, normalize=True) kullanılır; `True` ise `configClipVersion`/`configClipNormalize` alt config'leri açılır.
- `restart: Literal[True] = True` — model yükleme kararını etkileyen bir config değiştiğinde platformun `bootstrap()`'ı yeniden tetiklemesi için zorunlu.
- Kullanıldığı executor'lar: ilgili executor ✅

### 5.2 `ConfigClipVersion` / `ConfigPerceptionEncoderVersion`
```python
class ConfigClipVersion(Config):
    name: Literal["clipVersion"] = "ClipVersion"
    value: Literal["ViT-B-16", "ViT-B-32", "RN50"] = "ViT-B-16"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"
```
- CLIP seçenekleri: `ViT-B-16` (varsayılan), `ViT-B-32`, `RN50`
- Perception Encoder seçenekleri: `PE-Core-B16-224` (varsayılan), `PE-Core-L14-336`
- Yalnızca `ConfigXXXAdvance=True` iken erişilebilir

### 5.3 `ConfigClipNormalize` / `ConfigPerceptionEncoderNormalize`
```python
class ConfigClipNormalize(Config):
    name: Literal["clipNormalize"] = "ClipNormalize"
    value: bool = True
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"
```
- Varsayılan: `True` (kosinüs benzerliği için hazır)
- Yalnızca `ConfigXXXAdvance=True` iken erişilebilir

---

## 6. Çıktı (Output) Parametreleri

### 6.1 `OutputEmbedding`
```python
class OutputEmbedding(Output):
    name: Literal["outputEmbedding"] = "outputEmbedding"
    value: List[float]
    type: Literal["list"] = "list"
```

### 6.2 `OutputMeta`
```python
class OutputMeta(Output):
    name: Literal["outputMeta"] = "outputMeta"
    value: dict
    type: Literal["object"] = "object"
```
- Yapı örneği:
```json
{
  "model_family": "CLIP",
  "model_version": "ViT-B-16",
  "input_type": "image",
  "embedding_dim": 512
}
```
- Neden gerekli: Farklı model versiyonlarıyla üretilmiş embedding'ler AYNI vektör uzayında değildir; bu meta bilgi, yanlış kıyaslamaları önlemek içindir.

---

## 7. Veri Modelleri

### Response inşası (ASCII akış, `ObjectTracking/src/utils/response.py` ile aynı desen)

```
[Executor.run()]
      |
      V
OutputEmbedding(value=embedding) + OutputMeta(value=meta)
      |
      V
ClipEmbeddingOutputs(outputEmbedding=..., outputMeta=...)
      |
      V
ClipEmbeddingResponse(outputs=...)
      |
      V
ClipEmbeddingExecutor(value=clipResponse)
      |
      V
ConfigExecutor(value=clipExecutor)
      |
      V
PackageConfigs(executor=...)
      |
      V
PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
      |
      V
package.build_model(context)  --> gerçek Novavision Response nesnesi
```

⚠ Önemli fark: Response, Pydantic modelin `.dict()`'ini doğrudan dönerek DEĞİL, `PackageHelper.build_model(context)` çağrısıyla inşa edilir — bu, `context` (executor `self`) üzerinden gerekli platform-seviyesi alanları (`redis_db`, request meta bilgisi vb.) otomatik doldurur.

---

## 8. Metodoloji ve Algoritmalar

### 8.1 CLIP ile Embedding Çıkarımı (✅ Bu ortamda doğrulandı)

- Amaç: Görüntü veya metni `open_clip` üzerinden yüklenen CLIP modeliyle ortak vektör uzayına projekte etmek

- Adımlar:
  1. `open_clip.create_model_and_transforms(model_name, pretrained="openai")` ile model + preprocess yüklenir
  2. Görüntü: `preprocess(pil_image)` → `model.encode_image()`
  3. Metin: `tokenizer([text])` → `model.encode_text()`
  4. (Config'e göre) L2-normalizasyon: `features / features.norm(dim=-1, keepdim=True)`

- Doğrulama notu: Bu ortamda `open_clip_torch` kurulup rastgele başlatılmış ViT-B-16 ağırlığıyla forward-pass test edildi: görüntü embedding'i `(512,)` boyutunda üretildi, normalize sonrası norm ≈ 1.0 doğrulandı (gerçek OpenAI ağırlıkları bu ortamda indirilemedi — izinli domain listesinde huggingface.co/openaipublic yok — ama mimari/API akışı doğru çalışıyor).

- Avantajlar:
  - ✅ CPU'da çalışır, GPU şart değil
  - ✅ Zero-shot, ek eğitim gerektirmez

### 8.2 Perception Encoder ile Embedding Çıkarımı (⚠ Doğrulanmamış)

- Amaç: Meta'nın Perception Encoder modeliyle CLIP'e alternatif embedding üretmek
- ⚠ Bu ortamda `perception_models` paketi kurulu olmadığından import ve API çağrıları çalıştırılıp test edilememiştir.
- Gerçek ortamda önce:
  1. `pip install git+https://github.com/facebookresearch/perception_models.git`
  2. `core.vision_encoder.pe` içindeki gerçek sınıf/fonksiyon isimlerini `src/utils/utils.py::_load_perception_encoder` içinde teyit edin
  3. GPU/CUDA zorunluluğunu unutmayın — CPU'da çalışmaz
- Öneri: Önce `ClipEmbedding` ile pipeline'ı uçtan uca doğrulayıp, GPU erişimi olan bir makinede Perception Encoder tarafını ayrıca test etmek daha güvenli bir yol.

---

## 9. Claude Code İncelemesi Sonrası Düzeltmeler

Bu paket, Claude Code tarafından `cap-object-tracking` (ObjectTracking) referans
koduyla karşılaştırmalı olarak incelendi. Üç bulgu ve yapılan düzeltmeler:

### 9.1 (Kritik, düzeltildi) Config değerleri `request.get_param()` ile okunuyordu

**Sorun:** İlk versiyonda `self.normalize = self.request.get_param("ClipNormalize")`
satırı vardı. Referans kodda `request.get_param()` SADECE `inputs` altındaki
isimler için kullanılıyor (`"inputImage"`, `"inputDetections"`); config değerleri
ise her yerde ayrı bir mekanizmayla — `Application().get_param(config=config, name=...)`
ile ve SADECE `bootstrap(config: dict)` içinde — okunuyor. Referans repoda
`request.get_param()`'ın bir config adıyla çağrıldığı tek örnek yok.

**Risk:** Eğer gerçek SDK'da `get_param()` yalnızca `inputs` sözlüğünde arıyorsa,
bu çağrı sessizce `None` döner ve "Normalize Embedding" toggle'ı UI'da görünür
ama hiçbir zaman fiilen etkisi olmazdı.

**Düzeltme:** `normalize` artık SADECE `bootstrap()` içinde, `_build_clip_cfg()`
üzerinden `Application().get_param()` ile okunuyor ve `EmbeddingModelLoader`'a
bootstrap-time'da sabitleniyor (`self.normalize` attribute'u). `run()` içinde
`request.get_param()` artık yalnızca `"inputData"` (bir input) için çağrılıyor.

### 9.2 (Orta, düzeltildi) `restart: Literal[True]` eksikti

Referansta bootstrap()'ta model/tracker yükleyen her config zinciri
`restart: Literal[True] = True` taşıyan bir "Advance" sarmalayıcısıyla
korunuyor (ör. `ConfigBoTSortAdvance`). İlk versiyonda `ConfigClipVersion`
doğrudan config listesinde duruyordu, `restart` işareti yoktu — kullanıcı
versiyonu değiştirse platform bunun `bootstrap()`'ı yeniden tetiklemesi
gerektiğini bilemeyebilirdi.

### 9.3 (Düşük, düzeltildi) Config'ler Advance sarmalayıcısı olmadan üst seviyedeydi

Referanstaki 5 executor'ın 5'i de TÜM ayarlanabilir parametrelerini
(tek bir bool/enum olsa bile) bir `ConfigXXXAdvance` (`dependentDropdownlist`,
`restart=True`) sarmalayıcısının arkasına koyuyor. İlk versiyonda
`configClipVersion`/`configClipNormalize` bu sarmalayıcı olmadan doğrudan
`ClipEmbeddingConfigs`'in üst seviyesindeydi.

**Düzeltme:** `ConfigClipAdvance`/`ConfigPerceptionEncoderAdvance` eklendi
(True/False varyantları, `restart=True`), tutarlılık sağlandı ve bu aynı
zamanda §9.2'deki restart eksikliğini de çözdü.

### 9.4 Küçük not (temizlendi)

`load_clip_loader`/`load_perception_encoder_loader` fonksiyonları `cfg.normalize`
hesaplanıp da `EmbeddingModelLoader`'a hiç geçirilmiyordu (ölü kod). Artık
`normalize=cfg.normalize` olarak loader'a aktarılıyor ve `embed_image`/`embed_text`
içinde parametre verilmezse bu değer kullanılıyor. Bu davranış, `sdks` modülünü
mock'layarak izole test edildi: `normalize=False` → norm≈22.3 (normalize edilmemiş),
`normalize=True` → norm=1.0 (doğrulandı).

### Hâlâ doğrulanamayan noktalar

- `field="option"` (bool/enum config'ler için) tipi referansta doğrulanmış bir
  widget, ama Advance sarmalayıcısı OLMADAN kullanılan bir örneği referans kodda
  yoktu — artık bu paket de Advance sarmalayıcısı kullandığı için bu endişe
  ortadan kalktı.
- `isinstance(self.input_data, dict) and self.input_data.get("type") == "Image"`
  kontrolü, `get_param()`'ın input'lar için ham dict/list döndürdüğü varsayımına
  dayanıyor (referanstaki `input_detections[0]["imgUID"]` kullanımıyla tutarlı)
  — düşük risk, ama gerçek SDK ile ilk testte teyit edilmeli.
- Perception Encoder tarafı hâlâ doğrulanamadı (bkz. §8.2).

---

## 10. Gerçek Ortam Testi Bulgusu (QuickGELU)

İlk gerçek çalıştırmada (senin makinende, Claude Code üzerinden) şu uyarı
gözlemlendi: `"a red buoy on water"` metni için `ViT-B-16` ile 512-D embedding
başarıyla üretildi, ancak model config'i ile OpenAI'nin pretrained
ağırlıkları arasında bir "QuickGELU mismatch" uyarısı çıktı.

**Kök neden (doğrulandı):** `open_clip.get_pretrained_cfg("ViT-B-16", "openai")`
config'inde `quick_gelu: True` alanı var — yani OpenAI'nin orijinal CLIP
ağırlıkları (`ViT-B-16`, `ViT-B-32`, `RN50` — üçü de kontrol edildi) `QuickGELU`
aktivasyonuyla eğitilmiş. `create_model_and_transforms()` çağrısında
`force_quick_gelu=True` verilmezse model varsayılan (standart) `nn.GELU` ile
kurulur ve ağırlıklar yanlış aktivasyon fonksiyonuyla yüklenmiş olur.

**Risk:** Bu "zararsız" bir uyarı değil — forward pass hatasız çalışır ve
bir embedding üretir, ama üretilen embedding eğitim zamanındaki aktivasyon
fonksiyonundan sapar. Yani sonuç sessizce hatalı olabilir (çalışıyor
görünür ama kalitesi düşük olabilir).

**Doğrulama:** Bu ortamda katman tipi doğrudan incelendi:
```python
force_quick_gelu=False (varsayılan) -> nn.GELU
force_quick_gelu=True               -> nn.QuickGELU
```

**Düzeltme:** `src/utils/utils.py::EmbeddingModelLoader._load_clip()` ve
`apps/quick_test.py` içinde `force_quick_gelu=True` eklendi.
