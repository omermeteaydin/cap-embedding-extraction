# cap-embedding-extraction

A Novavision capsule package that produces image/text embeddings using CLIP
and Perception Encoder models. The structure is aligned with the real
conventions of the `cap-object-tracking` (ObjectTracking) package
(PackageModel, PackageHelper, Application().get_param(), Capsule/Executor
flow).

- Input: a single image **or** free text (`inputData`)
- Output: float embedding vector (`outputEmbedding`) + metadata (`outputMeta`)
- Two separate executors: `ClipEmbedding` and `PerceptionEncoderEmbedding`
- Detailed documentation: `DOCUMENTATION.md`

## Do I need a GPU?

| Method | GPU required? |
|---|---|
| CLIP (ViT-B-16 / ViT-B-32 / RN50) | No -- runs on CPU |
| Perception Encoder | Yes -- CUDA required |

If you don't have a GPU, **start with the CLIP executor**; the Perception
Encoder side is not verified yet (see the note below).

## Quick test (without the SDK, using open_clip directly)

```bash
pip install -r requirements.txt
python apps/quick_test.py --image resources/sample.jpg --version ViT-B-16
python apps/quick_test.py --text "a red buoy on water" --version ViT-B-16
```

## Real platform flow (with the SDK)

`apps/inference.py` -- same pattern as `ObjectTracking/apps/inference.py`,
sends a request to the HTTP endpoint through `PackageModel`.

## Things to know

- The `sdks.novavision` imports and the `PackageHelper.build_model()` /
  `Application().get_param()` / `Image.get_frame()` calls were modeled
  one-to-one on the real usage in the `ObjectTracking` (cap-object-tracking)
  package.
- The CLIP side is fully functional with `open_clip_torch`, and the
  forward-pass mechanics were verified in this environment (without
  downloading weights, using a randomly-initialized model).
- The Perception Encoder side depends on Meta's `perception_models` package
  and could not be tested in this environment; the
  `_load_perception_encoder` function in `src/utils/utils.py` was written
  as best-effort -- manual verification is required before using it in
  production.
- The appearance-extractor pattern in your own `BoTSORTTracker/reid.py`
  (crop -> batch -> normalize -> cache) was used as a reference.
