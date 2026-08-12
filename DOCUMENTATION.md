# EmbeddingExtraction - DOCUMENTATION

## 1. Overview

### Purpose of the package

The EmbeddingExtraction package is a capsule that produces semantic embedding vectors from images and/or free text using **CLIP** or **Perception Encoder** models. It is aligned with the real conventions of the `cap-object-tracking` (ObjectTracking) package (PackageModel/PackageHelper/Application().get_param()/Capsule-Executor flow). This package:

- Accepts a single image or text
- Exposes two separate executors: `ClipEmbedding` and `PerceptionEncoderEmbedding`
- Returns metadata alongside the output indicating which model/version produced it
- Applies optional L2 normalization

### Key features

- Semantic embedding extraction from both images and text
- CLIP versions: ViT-B-16, ViT-B-32, RN50 (runs on CPU, GPU not required)
- Perception Encoder versions: PE-Core-B16-224, PE-Core-L14-336 (GPU/CUDA required)
- L2 normalization (toggleable config)
- Nested config structure with `restart=True`, matching the "Advance" toggle pattern in `ObjectTracking` one-to-one (`ConfigClipAdvance` -> True/False variants, see sections 5.1 and 9.3)
- Model caching at bootstrap time (not reloaded on every `run()` call)
- Real Novavision response mechanism via `PackageHelper.build_model()`

### Supported classes / models / types

| ID | Name | Description |
|----|------|---------|
| 1  | `ClipEmbedding` | CLIP-based embedding executor -- `src/executors/ClipEmbedding.py` |
| 2  | `PerceptionEncoderEmbedding` | Perception Encoder-based embedding executor (GPU required, not verified) |
| 3  | `PackageModel` | Overall package structure definition (configs, executor) |
| 4  | `InputData` | Pydantic input model -- Image or free text (Union) |
| 5  | `EmbeddingModelLoader` | Model loading/caching/inference class (`utils/utils.py`) |
| 6  | `ConfigClipVersion` / `ConfigPerceptionEncoderVersion` | Model variant selection |
| 7  | `ConfigClipNormalize` / `ConfigPerceptionEncoderNormalize` | L2 normalization on/off |
| 8  | `OutputEmbedding` | Float embedding vector output |
| 9  | `OutputMeta` | Model family/version/dimension info |

---

## 2. Architecture and Technologies

### Technology stack
- Framework: Python 3.9+
- Model libraries: `open_clip_torch` (CLIP, verified), Meta `perception_models` (Perception Encoder, not verified)
- Image processing: Pillow, NumPy, OpenCV (via `sdks.novavision`)
- Deep learning: PyTorch (CPU or CUDA)
- API: `sdks.novavision` (Capsule, Executor, PackageHelper, Application, Image) -- same SDK usage as the `ObjectTracking` package

### Project structure (tree format)

```
cap-embedding-extraction/
├── LICENSE                              # MIT (DigiNova)
├── README.md
├── DOCUMENTATION.md                     # (this file)
├── setup.py                             # novavision.cap.embedding-extraction package dir
├── requirements.txt
├── .gitignore
├── __init__.py
├── apps/
│   ├── inference.py                     # Real platform HTTP client example (with SDK)
│   └── quick_test.py                    # Quick local test without SDK (open_clip directly)
├── resources/                           # Sample input images
├── src/
│   ├── __init__.py
│   ├── executors/
│   │   ├── ClipEmbedding.py             # CLIP executor
│   │   └── PerceptionEncoderEmbedding.py # Perception Encoder executor
│   ├── models/
│   │   └── PackageModel.py              # Pydantic models
│   └── utils/
│       ├── response.py                  # Response construction via PackageHelper
│       └── utils.py                     # EmbeddingModelLoader, config->cfg conversion
```

Notes:
- `ClipEmbedding.py` / `PerceptionEncoderEmbedding.py` -- each is its own `Capsule` subclass; `bootstrap()` loads the model once, `run()` produces an embedding per request. Same skeleton as `ObjectTracking/src/executors/BoTSortTracking.py`.
- `PackageModel.py` -- Pydantic models: a separate Request/Response/Executor triplet per executor, same as in `ObjectTracking`.
- `utils/utils.py` -- the `EmbeddingModelLoader` class plus `_build_clip_cfg`/`_build_perception_encoder_cfg` functions that read config via `Application().get_param()` (same pattern as `_build_bot_sort_cfg`).
- `utils/response.py` -- builds the real response object via `PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs).build_model(context)`.

---

## 3. Executors and Operating Modes

### `ClipEmbedding` (full path: `src/executors/ClipEmbedding.py`)

- Purpose: produce a semantic embedding from an image or text using CLIP.
- Use case:
  - Zero-shot visual/text similarity search
  - Extracting a coarse appearance signature on a detection crop
  - The only embedding method that can run on GPU-less machines (e.g. local dev environment)
- How it works (numbered steps):
  1. `bootstrap(config)` -> reads `ConfigClipAdvance` via `Application().get_param()`; if `True`, uses `ClipVersion`/`ClipNormalize`, if `False`, uses defaults; builds `EmbeddingModelLoader` (normalize is fixed on the loader at bootstrap time)
  2. `__init__` -> reads ONLY the input via `self.request.get_param("inputData")` (NOT config -- see section 9)
  3. `run()`:
     1. If the input is an image, the frame is fetched via `Image.get_frame(img=..., redis_db=self.redis_db)`
     2. If the input is text, it is processed directly as a string
     3. `loader.embed_image()` / `loader.embed_text()` is called (no normalize argument is passed; the loader's bootstrap-time normalize is used)
     4. The real Response is built via `PackageHelper` through `build_clip_response()`
- Key methods: `__init__`, `bootstrap(config)` (staticmethod), `run(self)`
- End of file: `if "__main__" == __name__: Executor(sys.argv[1]).run()`

### `PerceptionEncoderEmbedding` (full path: `src/executors/PerceptionEncoderEmbedding.py`)

- Purpose: produce an alternative embedding to CLIP using Meta's Perception Encoder.
- GPU/CUDA is required; **not verified end-to-end** in this environment because the `perception_models` package is not installed.
- How it works: identical flow to `ClipEmbedding`, only `load_perception_encoder_loader()` is called instead.

---

## 4. Input Parameters

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
- Definition: a single image or free text (Union type)
- Used by executors: ClipEmbedding, PerceptionEncoderEmbedding

---

## 5. Configuration Parameters

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
- Definition: same pattern as `ConfigBoTSortAdvance` in `ObjectTracking`, one-to-one. If `False`, default values are used (ViT-B-16, normalize=True); if `True`, the `configClipVersion`/`configClipNormalize` sub-configs are exposed.
- `restart: Literal[True] = True` -- required so the platform re-triggers `bootstrap()` when a config that affects the model-loading decision changes.
- Used by executors: the corresponding executor

### 5.2 `ConfigClipVersion` / `ConfigPerceptionEncoderVersion`
```python
class ConfigClipVersion(Config):
    name: Literal["clipVersion"] = "ClipVersion"
    value: Literal["ViT-B-16", "ViT-B-32", "RN50"] = "ViT-B-16"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"
```
- CLIP options: `ViT-B-16` (default), `ViT-B-32`, `RN50`
- Perception Encoder options: `PE-Core-B16-224` (default), `PE-Core-L14-336`
- Only accessible when `ConfigXXXAdvance=True`

### 5.3 `ConfigClipNormalize` / `ConfigPerceptionEncoderNormalize`
```python
class ConfigClipNormalize(Config):
    name: Literal["clipNormalize"] = "ClipNormalize"
    value: bool = True
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"
```
- Default: `True` (ready for cosine similarity)
- Only accessible when `ConfigXXXAdvance=True`

---

## 6. Output Parameters

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
- Example structure:
```json
{
  "model_family": "CLIP",
  "model_version": "ViT-B-16",
  "input_type": "image",
  "embedding_dim": 512
}
```
- Why it's needed: embeddings produced by different model versions do NOT live in the same vector space; this metadata exists to prevent incorrect comparisons.

---

## 7. Data Models

### Response construction (ASCII flow, same pattern as `ObjectTracking/src/utils/response.py`)

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
package.build_model(context)  --> real Novavision Response object
```

Important difference: the Response is NOT built by directly returning the Pydantic model's `.dict()` -- it's built via a `PackageHelper.build_model(context)` call, which automatically fills in the required platform-level fields (`redis_db`, request metadata, etc.) using `context` (the executor's `self`).

---

## 8. Methodology and Algorithms

### 8.1 Embedding Extraction with CLIP (verified in this environment)

- Purpose: project an image or text into a shared vector space using a CLIP model loaded via `open_clip`

- Steps:
  1. Load the model + preprocess pipeline via `open_clip.create_model_and_transforms(model_name, pretrained="openai")`
  2. Image: `preprocess(pil_image)` -> `model.encode_image()`
  3. Text: `tokenizer([text])` -> `model.encode_text()`
  4. (Depending on config) L2 normalization: `features / features.norm(dim=-1, keepdim=True)`

- Verification note: in this environment, `open_clip_torch` was installed and the forward pass was tested with a randomly-initialized ViT-B-16: the image embedding was produced with shape `(512,)`, and the norm after normalization was confirmed to be approximately 1.0 (the real OpenAI weights could not be downloaded in this environment -- huggingface.co/openaipublic is not on the allowlisted domain list -- but the architecture/API flow is confirmed correct).

- Advantages:
  - Runs on CPU, GPU not required
  - Zero-shot, no additional training required

### 8.2 Embedding Extraction with Perception Encoder (not verified)

- Purpose: produce an alternative embedding to CLIP using Meta's Perception Encoder model
- The `perception_models` package is not installed in this environment, so the import and API calls could not be run or tested.
- Before using in a real environment:
  1. `pip install git+https://github.com/facebookresearch/perception_models.git`
  2. Confirm the real class/function names inside `core.vision_encoder.pe` against `src/utils/utils.py::_load_perception_encoder`
  3. Remember GPU/CUDA is required -- it does not run on CPU
- Recommendation: verify the pipeline end-to-end with `ClipEmbedding` first, then separately test the Perception Encoder side on a machine with GPU access.

---

## 9. Fixes Made After Claude Code Review

This package was reviewed by Claude Code against the `cap-object-tracking` (ObjectTracking)
reference code. Three findings and the fixes applied:

### 9.1 (Critical, fixed) Config values were being read via `request.get_param()`

**Problem:** the first version had the line
`self.normalize = self.request.get_param("ClipNormalize")`. In the reference code,
`request.get_param()` is used ONLY for names under `inputs` (`"inputImage"`,
`"inputDetections"`); config values are read everywhere through a separate
mechanism -- `Application().get_param(config=config, name=...)` -- and ONLY
inside `bootstrap(config: dict)`. There is not a single example in the
reference repo of `request.get_param()` being called with a config name.

**Risk:** if the real SDK's `get_param()` only looks inside the `inputs` dict,
this call would silently return `None`, and the "Normalize Embedding" toggle
would appear in the UI but never actually have any effect.

**Fix:** `normalize` is now read ONLY inside `bootstrap()`, via
`_build_clip_cfg()` using `Application().get_param()`, and is fixed on
`EmbeddingModelLoader` at bootstrap time (the `self.normalize` attribute).
Inside `run()`, `request.get_param()` is now only called for `"inputData"`
(an input).

### 9.2 (Medium, fixed) `restart: Literal[True]` was missing

In the reference, every config chain that loads a model/tracker inside
bootstrap() is protected by an "Advance" wrapper carrying
`restart: Literal[True] = True` (e.g. `ConfigBoTSortAdvance`). In the first
version, `ConfigClipVersion` sat directly in the config list with no `restart`
flag -- if the user changed the version, the platform might not know it
needed to re-trigger `bootstrap()`.

### 9.3 (Low, fixed) Configs sat at the top level without an Advance wrapper

All 5 executors in the reference put ALL of their configurable parameters
(even a single bool/enum) behind a `ConfigXXXAdvance`
(`dependentDropdownlist`, `restart=True`) wrapper. In the first version,
`configClipVersion`/`configClipNormalize` sat directly at the top level of
`ClipEmbeddingConfigs` without this wrapper.

**Fix:** `ConfigClipAdvance`/`ConfigPerceptionEncoderAdvance` were added
(True/False variants, `restart=True`), bringing consistency and also fixing
the missing-restart issue from section 9.2 at the same time.

### 9.4 Minor note (cleaned up)

`load_clip_loader`/`load_perception_encoder_loader` computed `cfg.normalize`
but never passed it to `EmbeddingModelLoader` (dead code). Now
`normalize=cfg.normalize` is passed to the loader and is used in
`embed_image`/`embed_text` whenever no parameter is given at call time. This
behavior was verified in isolation by mocking the `sdks` module:
`normalize=False` -> norm ≈ 22.3 (not normalized), `normalize=True` -> norm =
1.0 (confirmed).

### Points still unverified

- The `field="option"` type (for bool/enum configs) is a widget type
  confirmed in the reference, but the reference code had no example of it
  being used WITHOUT an Advance wrapper -- this concern is now moot since
  this package also uses an Advance wrapper.
- The `isinstance(self.input_data, dict) and self.input_data.get("type") ==
  "Image"` check relies on the assumption that `get_param()` returns a raw
  dict/list for inputs (consistent with the reference's
  `input_detections[0]["imgUID"]` usage) -- low risk, but should be
  confirmed on the first real SDK test.
- The Perception Encoder side is still unverified (see section 8.2).

---

## 10. Real-Environment Test Finding (QuickGELU)

On the first real run (on your machine, via Claude Code), the following
warning was observed: a 512-D embedding was successfully produced for the
text `"a red buoy on water"` with `ViT-B-16`, but a "QuickGELU mismatch"
warning appeared between the model config and OpenAI's pretrained weights.

**Root cause (verified):** `open_clip.get_pretrained_cfg("ViT-B-16",
"openai")`'s config has a `quick_gelu: True` field -- meaning OpenAI's
original CLIP weights (`ViT-B-16`, `ViT-B-32`, `RN50` -- all three checked)
were trained with QuickGELU activation. Without `force_quick_gelu=True` in
the `create_model_and_transforms()` call, the model is built with the
default (standard) `nn.GELU`, and the weights end up loaded with the wrong
activation function.

**Risk:** this is not a "harmless" warning -- the forward pass runs without
error and produces an embedding, but the resulting embedding drifts from
the activation function used at training time. In other words, the result
can be silently wrong (it appears to work, but quality may be degraded).

**Verification:** the layer type was directly inspected in this environment:
```python
force_quick_gelu=False (default) -> nn.GELU
force_quick_gelu=True            -> nn.QuickGELU
```

**Fix:** `force_quick_gelu=True` was added in
`src/utils/utils.py::EmbeddingModelLoader._load_clip()` and in
`apps/quick_test.py`.
