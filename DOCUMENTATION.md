# EmbeddingExtraction - DOCUMENTATION

## 1. Overview

### What this package does

The EmbeddingExtraction package generates semantic embedding vectors from
images and/or free text using **CLIP** or **Perception Encoder** models,
and provides a CLIP-based zero-shot comparison mode. It exposes **three
executors**, following the same one-task-per-executor pattern used in
`ObjectTracking` (SORT / OC-SORT / ByteTrack / BoT-SORT):

- **ClipGenerate** — image or text -> a single embedding vector
- **ClipComparison** — image + a list of text labels -> a similarity score per label (zero-shot classification)
- **PerceptionEncoder** — image or text -> a single embedding vector (Generate only; no comparison mode, see rationale below)

### Why Generate and Comparison are separate executors

Generate and Comparison have fundamentally different input/output
contracts:

| | Input | Output |
|---|---|---|
| Generate | 1 image OR 1 text | 1 embedding vector |
| Comparison | 1 image + N text labels | N similarity scores |

Rather than cramming both shapes into a single Request/Response via a
config-driven "mode" switch, each is exposed as its own Task option in
`ConfigExecutor` -- mirroring how `ObjectTracking` exposes SORT, OC-SORT,
ByteTrack, and BoT-SORT as four separate executors inside one package.
This is a proven, platform-supported pattern.

### Why Perception Encoder has no Comparison mode

Roboflow does not currently offer a dedicated Perception Encoder
comparison block (only a `Perception Encoder Embedding Model` block for
generation, plus the model-agnostic `Cosine Similarity` block for
comparing any two embeddings). This package mirrors that: Perception
Encoder only exposes a Generate-equivalent executor.

### Key features

- ✅ Image and text embedding generation (CLIP, Perception Encoder)
- ✅ CLIP versions: ViT-B-16, ViT-B-32, RN50 (run on CPU, GPU not required)
- ✅ Perception Encoder versions: PE-Core-B16-224, PE-Core-L14-336 (⚠ GPU/CUDA required)
- ✅ Zero-shot image-to-text-labels comparison (ClipComparison), mirrors Roboflow's `roboflow_core/clip_comparison@v2`
- ✅ L2-normalization toggle for Generate executors
- ✅ Advance-toggle config pattern consistent with `ObjectTracking` (`restart=True` where the config affects model loading)
- ✅ Model caching at bootstrap-time (not reloaded on every `run()` call)
- ✅ Real Response construction via `PackageHelper.build_model()`

### Supported classes / models / types

| ID | Name | Description |
|----|------|-------------|
| 1  | `ClipGenerate` | CLIP embedding generation executor — `src/executors/ClipGenerate.py` |
| 2  | `ClipComparison` | CLIP zero-shot image/text-labels comparison executor — `src/executors/ClipComparison.py` |
| 3  | `PerceptionEncoder` | Perception Encoder embedding generation executor (⚠ GPU required, unverified) |
| 4  | `PackageModel` | Top-level package structure (configs, executor) |
| 5  | `InputData` | Pydantic input model for Generate executors — Image or free text (Union) |
| 6  | `InputComparisonImage` / `InputComparisonClasses` | Pydantic input models for ClipComparison — image and text-label list |
| 7  | `EmbeddingModelLoader` | Model loading / caching / inference class (`utils/utils.py`) — also implements `compare_image_to_texts` |
| 8  | `OutputEmbedding` | Float embedding vector output (Generate executors) |
| 9  | `OutputSimilarities` | Per-label similarity score output (ClipComparison) |
| 10 | `OutputMeta` | Model family / version / traceability info |

---

## 2. Architecture and Technologies

### Technology stack
- Framework: Python 3.9+
- Model libraries: `open_clip_torch` (CLIP, verified), Meta `perception_models` (Perception Encoder, ⚠ unverified)
- Image handling: Pillow, NumPy, OpenCV (via `sdks.novavision`)
- Deep learning: PyTorch (CPU or CUDA)
- API: `sdks.novavision` (Capsule, Executor, PackageHelper, Application, Image) — same SDK usage as `ObjectTracking`

### Project structure (tree)

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
│   ├── inference.py                     # Real platform HTTP client examples (SDK-based)
│   └── quick_test.py                    # SDK-free local sanity check (raw open_clip)
├── resources/                           # Example input images
├── src/
│   ├── __init__.py
│   ├── executors/
│   │   ├── ClipGenerate.py              # CLIP embedding generation executor
│   │   ├── ClipComparison.py            # CLIP zero-shot comparison executor
│   │   └── PerceptionEncoder.py         # Perception Encoder generation executor
│   ├── models/
│   │   └── PackageModel.py              # Pydantic models
│   └── utils/
│       ├── response.py                  # Response construction via PackageHelper
│       └── utils.py                     # EmbeddingModelLoader, config -> cfg resolution
```

---

## 3. Executors and Modes of Operation

### `ClipGenerate` (`src/executors/ClipGenerate.py`)

- Purpose: produce a semantic embedding from an image or text using CLIP.
- Use cases:
  - ✅ Zero-shot visual/textual similarity search
  - ✅ Run on a detection crop to produce a coarse appearance signature
  - ✅ Only embedding method usable without a GPU
- Flow:
  1. `bootstrap(config)` reads `ConfigClipGenerateAdvance` via `Application().get_param()`; if `True`, reads `ClipGenerateVersion`/`ClipGenerateNormalize`, else uses defaults; builds `EmbeddingModelLoader`
  2. `__init__` reads `inputData` via `self.request.get_param("inputData")`
  3. `run()`: if input is an image, `Image.get_frame(...)` then `loader.embed_image()`; if text, `loader.embed_text()`; response built via `build_clip_generate_response()`

### `ClipComparison` (`src/executors/ClipComparison.py`)

- Purpose: zero-shot classification — compare one image against a list of
  free-text labels and return a similarity score per label.
- Use cases:
  - ✅ Classify an image without training a dedicated model (e.g. "is this NSFW", "what type of vessel is this")
- Flow:
  1. `bootstrap(config)` reads `ConfigClipComparisonAdvance` (same Advance-toggle pattern, no `normalize` toggle — comparison always uses normalized embeddings)
  2. `__init__` reads `inputImage` and `inputClasses` separately (two inputs, same pattern as `BoTSortTracking`'s `inputImage`/`inputDetections`)
  3. `run()`: `Image.get_frame(...)` then `loader.compare_image_to_texts(image_array, classes)` — embeds the image once, embeds each label, returns cosine similarity per label as a dict

### `PerceptionEncoder` (`src/executors/PerceptionEncoder.py`)

- Purpose: produce a semantic embedding from an image or text using Meta's Perception Encoder.
- ⚠ GPU/CUDA required; `perception_models` package not installed in this environment, so this path is **unverified end-to-end**.
- Flow: identical to `ClipGenerate`, calling `load_perception_encoder_loader()` instead.

---

## 4. Input Parameters

### 4.1 `InputData` (ClipGenerate / PerceptionEncoder)
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
```
- A single image OR free text (Union type).
- Used by: ClipGenerate ✅, PerceptionEncoder ✅

### 4.2 `InputComparisonImage` / `InputComparisonClasses` (ClipComparison)
```python
class InputComparisonImage(Input):
    name: Literal["inputImage"] = "inputImage"
    value: Image
    type: Literal["object"] = "object"

class InputComparisonClasses(Input):
    name: Literal["inputClasses"] = "inputClasses"
    value: List[str] = Field(default_factory=list)
    type: Literal["list"] = "list"
```
- `inputImage`: the image to classify.
- `inputClasses`: free-text labels to compare against, e.g. `["boat", "buoy", "obstacle"]`.
- Used by: ClipComparison ✅

---

## 5. Configuration Parameters

### 5.1 `ConfigClipGenerateAdvance` / `ConfigClipComparisonAdvance` / `ConfigPerceptionEncoderAdvance`

Same nested True/False pattern as `ConfigBoTSortAdvance` in `ObjectTracking`:
```python
class ConfigClipGenerateAdvanceTrue(Config):
    name: Literal["True"] = "True"
    value: Literal["True"] = "True"
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"
    configClipGenerateVersion: ConfigClipGenerateVersion
    configClipGenerateNormalize: ConfigClipGenerateNormalize

class ConfigClipGenerateAdvance(Config):
    name: Literal["ConfigClipGenerateAdvance"] = "ConfigClipGenerateAdvance"
    value: Union[ConfigClipGenerateAdvanceTrue, ConfigClipGenerateAdvanceFalse]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"
    restart: Literal[True] = True
```
- `restart: Literal[True] = True` is mandatory here because the sub-configs
  (model version) affect what `bootstrap()` loads.
- `ConfigClipComparisonAdvance` follows the exact same pattern but only
  nests `configClipComparisonVersion` (no `normalize` toggle — see §3).

### 5.2 `ConfigClipGenerateVersion` / `ConfigClipComparisonVersion` / `ConfigPerceptionEncoderVersion`
```python
class ConfigClipGenerateVersion(Config):
    name: Literal["clipGenerateVersion"] = "ClipGenerateVersion"
    value: Literal["ViT-B-16", "ViT-B-32", "RN50"] = "ViT-B-16"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"
```
- CLIP options: `ViT-B-16` (default), `ViT-B-32`, `RN50`
- Perception Encoder options: `PE-Core-B16-224` (default), `PE-Core-L14-336`
- Only reachable when the corresponding `ConfigXXXAdvance=True`

### 5.3 `ConfigClipGenerateNormalize` / `ConfigPerceptionEncoderNormalize`
```python
class ConfigClipGenerateNormalize(Config):
    name: Literal["clipGenerateNormalize"] = "ClipGenerateNormalize"
    value: bool = True
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"
```
- Default: `True` (ready for cosine-similarity comparisons).
- ClipComparison has no equivalent toggle: `compare_image_to_texts()`
  always L2-normalizes both sides internally, because cosine similarity
  requires unit vectors to reduce to a plain dot product — leaving this
  configurable would risk producing mathematically incorrect scores.

---

## 6. Output Parameters

### 6.1 `OutputEmbedding` (ClipGenerate / PerceptionEncoder)
```python
class OutputEmbedding(Output):
    name: Literal["outputEmbedding"] = "outputEmbedding"
    value: List[float]
    type: Literal["list"] = "list"
```

### 6.2 `OutputSimilarities` (ClipComparison)
```python
class OutputSimilarities(Output):
    name: Literal["outputSimilarities"] = "outputSimilarities"
    value: Dict[str, float]
    type: Literal["object"] = "object"
```
- Example: `{"boat": 0.82, "buoy": 0.41, "obstacle": 0.09}`
- Values are cosine similarities (-1 to 1) between the image embedding
  and each label's text embedding.

### 6.3 `OutputMeta` (all executors)
```python
class OutputMeta(Output):
    name: Literal["outputMeta"] = "outputMeta"
    value: dict
    type: Literal["object"] = "object"
```
- ClipGenerate / PerceptionEncoder: `{"model_family", "model_version", "input_type", "embedding_dim"}`
- ClipComparison: `{"model_family", "model_version", "num_classes"}`
- Why needed: embeddings produced by different model versions live in
  different vector spaces and must never be silently compared.

---

## 7. Data Models

### Response construction (ASCII flow, same pattern as `ObjectTracking/src/utils/response.py`)

```
[Executor.run()]
      |
      V
OutputXxx(value=...) + OutputMeta(value=meta)
      |
      V
XxxOutputs(...)
      |
      V
XxxResponse(outputs=...)
      |
      V
XxxExecutor(value=xxxResponse)
      |
      V
ConfigExecutor(value=xxxExecutor)
      |
      V
PackageConfigs(executor=...)
      |
      V
PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
      |
      V
package.build_model(context)  --> real NovaVision Response object
```

⚠ Note: the response is built via `PackageHelper.build_model(context)`,
not by returning a Pydantic model's `.dict()` directly — this populates
platform-level fields (`redis_db`, request metadata, etc.) from `context`
(the executor's `self`).

---

## 8. Methodology and Algorithms

### 8.1 CLIP Embedding Generation (✅ verified in this environment)

- Steps:
  1. `open_clip.create_model_and_transforms(model_name, pretrained="openai", force_quick_gelu=True)` loads the model + preprocessing pipeline
  2. Image: `preprocess(pil_image)` -> `model.encode_image()`
  3. Text: `tokenizer([text])` -> `model.encode_text()`
  4. (Config-dependent) L2-normalization: `features / features.norm(dim=-1, keepdim=True)`
- Verification: forward-pass tested in this environment with a randomly
  initialized ViT-B-16 (real OpenAI weights could not be downloaded here
  — huggingface.co/openaipublic are not in the allowed network domain
  list), confirming a `(512,)`-shaped output and norm ≈ 1.0 after
  normalization. `force_quick_gelu` behavior was independently verified
  by inspecting the layer type (`nn.GELU` vs `nn.QuickGELU`).

### 8.2 CLIP Zero-Shot Comparison (✅ mechanics verified in this environment)

- Purpose: classify an image against free-text labels without training a
  dedicated classifier — mirrors Roboflow's `roboflow_core/clip_comparison@v2`.
- Steps:
  1. Embed the image once via `encode_image()`, L2-normalize
  2. Embed all provided labels in a single batched `encode_text()` call, L2-normalize
  3. Cosine similarity reduces to a dot product once both sides are unit vectors: `image_features @ text_features.T`
  4. Zip label strings to their similarity scores into a `Dict[str, float]`
- Verification: `compare_image_to_texts()` was tested in this environment
  with a randomly initialized ViT-B-16 and three labels (`"boat"`,
  `"buoy"`, `"obstacle"`); confirmed all scores fall within [-1, 1] and
  an empty label list returns an empty dict without error.

### 8.3 Perception Encoder Embedding Generation (⚠ Unverified)

- Same as §8.1 but using `core.vision_encoder.pe.CLIP.from_config(...)`.
- ⚠ The `perception_models` package is not installed in this environment,
  so these import/API calls have **not** been executed or tested. Before
  going to production:
  1. `pip install git+https://github.com/facebookresearch/perception_models.git`
  2. Confirm the actual class/function names in `core.vision_encoder.pe`
     against the installed package version inside
     `src/utils/utils.py::_load_perception_encoder`
  3. Remember the GPU/CUDA requirement — this path does not run on CPU
 
  ---

## 9. Platform Integration Notes

This section documents findings from deploying and testing the package on
the real NovaVision platform, separate from the code-level implementation
covered in §1-8.

### 9.1 Flow was silently disabled

The Main flow was found to be in a "Disabled" state (visible only by
hovering over an icon next to the flow name in the Flows panel). This
alone explained why several early debug attempts (raising a deliberate
exception, swapping the `Capsule` base class for `Component`, stripping
all dependencies) produced no output and no LOG entries at all — the
container/environment never came up in the first place. Enabling it
(Set Environment > Web/Media > Edge) made the flow actually execute.

### 9.2 Platform LOG is unreliable — use outputMeta instead

Even with the flow enabled, the platform's LOG screen only ever showed an
unrelated WebSocket/certificate error
(`wss://cpu.api.novavision.ai:7002`, `ERR_CERT_DATE_INVALID`), never the
real Python traceback. To work around this, `ClipGenerate.bootstrap()`
and `.run()` were temporarily wrapped in try/except blocks that write the
exception type, message, and traceback (split into a line list, not one
long string — a long string was misinterpreted by the platform UI as
base64 image data and rendered as an "Image Preview" instead of text)
into `outputMeta`. This surfaced the real error directly in the flow's
Output/Raw tab without depending on LOG streaming.

### 9.3 Root cause: missing `open_clip_torch` dependency

The real error, caught via the above technique:
```
ModuleNotFoundError: No module named 'open_clip'
```

Investigation showed the platform does **not** install dependencies from
the package repo's `setup.py` or `requirements.txt` — this was later
confirmed by an internal team-wide announcement ("setup.py dosyaları
şu anda işlevsizdir. Gerekli paketler ... ilgili image dosyasındaki
requirements.txt'ye eklenmesi gerekir"). Instead, the platform builds
containers from a **separate** image repository
(`novavision-ai/image-pytorch`), whose own `requirements.prod` /
`requirements.dev` files did not list `open_clip_torch` (and did not
explicitly pin `torch` either — it was only pulled in transitively via
`ultralytics`).

Fix applied: a `develop-embedding` branch was opened from `develop` in
`image-pytorch`, adding to both `requirements.prod` and
`requirements.dev`:
```
torch==2.13.0
open_clip_torch==3.3.0
```
Pushed to `https://github.com/novavision-ai/image-pytorch/tree/develop-embedding`.

### 9.4 Container image size

Measured directly from PyPI package metadata (not estimated): the full
dependency chain for GPU-enabled `torch` (its default PyPI distribution
on Linux pulls in CUDA toolkit components — cuDNN, NCCL, etc.) totals
approximately **1.6 GB**. This is required because the package's
`PerceptionEncoder` executor needs GPU/CUDA; the CLIP-only path
(`ClipGenerate`, `ClipComparison`) does not require a GPU and could use a
CPU-only `torch` build (~200-500 MB) if Perception Encoder support were
ever dropped or split into a separate image.

### 9.5 Detection-crop flow — a ready-made "Crop" component exists

Hocam's stated priority (detection-based embedding before whole-image
embedding, see the original meeting transcript) does not require any
additional cropping code in this package. Roboflow's own architecture
(`Object Detection → Dynamic Crop → CLIP`, as separate chained blocks)
was mirrored: the platform's component library already includes a
`Crop` component with a `Dynamic` task (alongside `Absolute`,
`Relative`, `ROI`), which takes `image` + `detections` as input and
outputs a cropped image per detection — functionally equivalent to
Roboflow's `Dynamic Crop` block. No new capsule was needed.

Working flow topology (verified in the platform's flow editor, not yet
executed end-to-end due to the blocker in §9.6):
```
Image Load --(image)--> Yolo Inference --(image)--> [passthrough]
Image Load --(image)--> Crop.image
Yolo Inference --(detections)--> Crop.detections
Crop (Task: Dynamic) --(crop output)--> ClipGenerate.data
```

### 9.6 Current blocker: container build/registry failure

After adding the dependency (§9.3) and registering a new platform Image
record (Name: `EmbeddingPytorch`, pointing at the `image-pytorch`
`develop-embedding` branch), starting the container fails with the
container status flipping to **"Deleted"** and the following error
visible in the Docker manager log:
```
DockerManager - An error occurred while creating the Docker container:
404 Client Error for http+docker://localhost/v1.55/images/create?...:
Not Found ("pull access denied for
omermeteaydin-workspace-embeddingpytorch, repository does not exist or
may require 'docker login'")
```

This indicates the platform expects to `docker pull` a pre-built image
from a registry, rather than building one directly from the GitHub repo
at container-start time. No such image currently exists in the
registry — the "Create Image" form in the platform only registers
metadata (name, repo URL, branch) and does not itself trigger a
build-and-push step. Whether that build/push is triggered automatically,
requires a manual step, or requires the source repo/image to be public
is unresolved as of this writing; escalated to Uğur Yıldız
(`@ugur.yildiz`) for clarification. **No embedding output has been
successfully produced on the real platform yet** — all verification to
date (§8) was performed with randomly-initialized weights in an isolated
environment, not with real downloaded weights on the target platform.
