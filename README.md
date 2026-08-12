# cap-embedding-extraction

A Novavision capsule package that produces image/text embeddings using CLIP
and Perception Encoder models, plus a CLIP-based zero-shot comparison mode.
The structure is aligned with the real conventions of the
`cap-object-tracking` (ObjectTracking) package (PackageModel, PackageHelper,
Application().get_param(), Capsule/Executor flow).

> **Note:** The official repo has moved to
> [`novavision-ai/cap-embedding`](https://github.com/novavision-ai/cap-embedding).
> This repo will be migrated there; until then it remains the active
> working copy.

## Three executors

| Executor | Input | Output |
|---|---|---|
| `ClipGenerate` | 1 image or text | 1 embedding vector |
| `ClipComparison` | 1 image + N text labels | N similarity scores (zero-shot classification) |
| `PerceptionEncoder` | 1 image or text | 1 embedding vector |

Generate and Comparison are separate executors (not a single "mode" toggle)
because their input/output shapes are fundamentally different -- see
`DOCUMENTATION.md` section 1 for the full rationale. Perception Encoder has
no comparison mode because Roboflow doesn't offer one either.

Detailed documentation: `DOCUMENTATION.md`

## Do I need a GPU?

| Method | GPU required? |
|---|---|
| CLIP (ViT-B-16 / ViT-B-32 / RN50) -- Generate & Comparison | No -- runs on CPU |
| Perception Encoder | Yes -- CUDA required |

If you don't have a GPU, **start with the CLIP executors**; the Perception
Encoder side is not verified yet (see the note below).

## Quick test (without the SDK, using open_clip directly)

```bash
pip install -r requirements.txt
python apps/quick_test.py --image resources/sample.jpg --version ViT-B-16
python apps/quick_test.py --text "a red buoy on water" --version ViT-B-16
```

## Real platform flow (with the SDK)

`apps/inference.py` -- same pattern as `ObjectTracking/apps/inference.py`,
sends a request to the HTTP endpoint through `PackageModel`. Includes
examples for all three executors (`inference_generate_image`,
`inference_generate_text`, `inference_comparison`).

## Things to know

- The `sdks.novavision` imports and the `PackageHelper.build_model()` /
  `Application().get_param()` / `Image.get_frame()` calls were modeled
  one-to-one on the real usage in the `ObjectTracking` (cap-object-tracking)
  package.
- The CLIP side is fully functional with `open_clip_torch`. Both
  `embed_image`/`embed_text` (Generate) and `compare_image_to_texts`
  (Comparison) mechanics were verified in this environment (without
  downloading real weights, using a randomly-initialized model).
- The Perception Encoder side depends on Meta's `perception_models` package
  and could not be tested in this environment; the
  `_load_perception_encoder` function in `src/utils/utils.py` was written
  as best-effort -- manual verification is required before using it in
  production.
- The appearance-extractor pattern in your own `BoTSORTTracker/reid.py`
  (crop -> batch -> normalize -> cache) was used as a reference.
