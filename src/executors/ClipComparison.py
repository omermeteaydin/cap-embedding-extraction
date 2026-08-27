"""Executor for CLIP-based zero-shot image/text comparison in the
NovaVision pipeline.

Takes a single image and a list of free-text class labels, and returns
a similarity score for each label -- mirrors Roboflow's
`roboflow_core/clip_comparison@v2` block. Useful for classifying images
without training a dedicated model (e.g. "is this image NSFW", "what
type of vessel is this").
"""

import os
import sys
import traceback

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.base.capsule import Capsule
from sdks.novavision.src.helper.executor import Executor
from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.application import Application

from capsules.EmbeddingExtraction.src.models.PackageModel import PackageModel
from capsules.EmbeddingExtraction.src.utils.response import build_clip_comparison_response
from capsules.EmbeddingExtraction.src.utils.utils import load_clip_comparison_loader

DEFAULT_CLASSES = "buoy,boat,person,building,water,sky,dog,mountain"

# Maps the dependentDropdownlist discriminator (see ConfigClipComparisonClasses
# in PackageModel.py) to the actual comma-separated label string.
CLASSES_PRESETS = {
    "Preset_Buoy": "buoy,boat,person,building,water,sky,dog,mountain",
    "Preset_Dock": "boat,buoy,dock,person,horizon,obstacle",
    "Preset_Vessel": "vessel,buoy,shore,water,sky",
}

# TRACE DEBUG (per team request): the previous try/except-only debug prints
# only fire on an exception, so when the platform never invokes run() at
# all -- no error, no output, total silence -- we had nothing to look at.
# These unconditional line-by-line prints run every time regardless of
# whether anything fails, so the container logs show exactly how far
# execution actually gets: is the module even imported? is __init__
# reached? is bootstrap() reached/finished? is run() reached at all?
# Remove all of these once we've confirmed where execution stops (or once
# we get real output and this whole investigation is closed out).
print("[TRACE] ClipComparison module - imported/loaded", file=sys.stderr, flush=True)


class ClipComparison(Capsule):
    def __init__(self, request, bootstrap):
        print("[TRACE] ClipComparison.__init__ - START", file=sys.stderr, flush=True)
        super().__init__(request, bootstrap)
        print("[TRACE] ClipComparison.__init__ - super().__init__ done", file=sys.stderr, flush=True)
        self.request.model = PackageModel(**(self.request.data))
        print("[TRACE] ClipComparison.__init__ - PackageModel(**request.data) built OK", file=sys.stderr, flush=True)
        self.input_image = self.request.get_param("inputImage")
        print(f"[TRACE] ClipComparison.__init__ - inputImage param retrieved, type={type(self.input_image)}", file=sys.stderr, flush=True)
        print("[TRACE] ClipComparison.__init__ - END", file=sys.stderr, flush=True)

    @staticmethod
    def bootstrap(config: dict) -> dict:
        print("[TRACE] ClipComparison.bootstrap - START", file=sys.stderr, flush=True)
        # TEMPORARY DEBUG: same reasoning as ClipGenerate.bootstrap -- the
        # platform LOG only shows a bare exception message (no traceback,
        # e.g. just "'value'" for a KeyError), so bootstrap failures are
        # caught here and handed to run() via the returned dict instead of
        # raising -- raising here would prevent run() (and outputMeta) from
        # ever being reached.
        # Remove this try/except once platform logging is confirmed reliable.
        try:
            print("[TRACE] ClipComparison.bootstrap - calling load_clip_comparison_loader", file=sys.stderr, flush=True)
            loader = load_clip_comparison_loader(config=config)
            print("[TRACE] ClipComparison.bootstrap - load_clip_comparison_loader returned OK", file=sys.stderr, flush=True)
            # Classes UI field removed from PackageModel.py (see the
            # comment on ClipComparisonConfigs there) -- it was declared
            # REQUIRED but the platform never rendered/sent it, which was
            # crashing PackageModel(**self.request.data) in __init__
            # before this code ever ran. Falling back to the fixed
            # DEFAULT_CLASSES list for now so we can confirm the node
            # actually produces output at all. CLASSES_PRESETS kept for
            # when we re-attempt a working UI field for this.
            classes_raw = DEFAULT_CLASSES
            print(f"[TRACE] ClipComparison.bootstrap - classes_raw={classes_raw!r}", file=sys.stderr, flush=True)
            result = {
                "loader": loader,
                "classes_raw": classes_raw,
                "bootstrap_error": None,
            }
            print("[TRACE] ClipComparison.bootstrap - RETURNING success result", file=sys.stderr, flush=True)
            return result
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"[ClipComparison.bootstrap] DEBUG ERROR:\n{tb}", file=sys.stderr, flush=True)
            print("[TRACE] ClipComparison.bootstrap - RETURNING error result", file=sys.stderr, flush=True)
            return {
                "loader": None,
                "classes_raw": DEFAULT_CLASSES,
                "bootstrap_error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": tb,
                },
            }

    def run(self):
        print("[TRACE] ClipComparison.run - START", file=sys.stderr, flush=True)
        # TEMPORARY DEBUG: same reasoning as ClipGenerate.run -- surfaces the
        # real exception (type + message + traceback) inside outputMeta so it
        # is visible in the flow's Output/Raw tab without depending on
        # platform LOG streaming. The traceback is split into a list of
        # lines (rather than one long string) because the platform's UI
        # misinterpreted a long single string as base64 image data and
        # rendered an "Image Preview" instead of the text.
        # Remove this try/except once platform LOG is confirmed reliable.
        bootstrap_error = self.bootstrap.get("bootstrap_error")
        print(f"[TRACE] ClipComparison.run - bootstrap_error={bootstrap_error}", file=sys.stderr, flush=True)
        if bootstrap_error:
            print("[TRACE] ClipComparison.run - bootstrap_error present, returning early with empty similarities", file=sys.stderr, flush=True)
            meta = {
                "DEBUG_STAGE": "bootstrap",
                "DEBUG_ERROR_TYPE": bootstrap_error["type"],
                "DEBUG_ERROR_MESSAGE": bootstrap_error["message"],
                "DEBUG_TRACEBACK_LINES": bootstrap_error["traceback"].splitlines(),
            }
            return build_clip_comparison_response(context=self, similarities={}, meta=meta)

        try:
            print("[TRACE] ClipComparison.run - reading loader from self.bootstrap", file=sys.stderr, flush=True)
            loader = self.bootstrap["loader"]
            print(f"[TRACE] ClipComparison.run - loader={loader}", file=sys.stderr, flush=True)

            print("[TRACE] ClipComparison.run - calling Image.get_frame", file=sys.stderr, flush=True)
            image = Image.get_frame(img=self.input_image, redis_db=self.redis_db)
            print(f"[TRACE] ClipComparison.run - Image.get_frame returned, image is None: {image is None}", file=sys.stderr, flush=True)
            image_array = image.value if image is not None else None
            print(f"[TRACE] ClipComparison.run - image_array is None: {image_array is None}", file=sys.stderr, flush=True)
            classes_raw = self.bootstrap.get("classes_raw") or DEFAULT_CLASSES
            classes = [label.strip() for label in classes_raw.split(",") if label.strip()]
            print(f"[TRACE] ClipComparison.run - classes={classes}", file=sys.stderr, flush=True)

            print("[TRACE] ClipComparison.run - calling loader.compare_image_to_texts", file=sys.stderr, flush=True)
            similarities = loader.compare_image_to_texts(image_array, classes)
            print(f"[TRACE] ClipComparison.run - similarities={similarities}", file=sys.stderr, flush=True)

            meta = {
                "model_family": loader.model_family,
                "model_version": loader.model_version,
                "num_classes": len(classes),
                "classes_raw": classes_raw,
            }
            print(f"[TRACE] ClipComparison.run - meta built: {meta}", file=sys.stderr, flush=True)

            print("[TRACE] ClipComparison.run - calling build_clip_comparison_response", file=sys.stderr, flush=True)
            packageModel = build_clip_comparison_response(
                context=self,
                similarities=similarities,
                meta=meta,
            )
            print("[TRACE] ClipComparison.run - build_clip_comparison_response OK, RETURNING", file=sys.stderr, flush=True)
            return packageModel
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"[ClipComparison.run] DEBUG ERROR:\n{tb}", file=sys.stderr, flush=True)
            print("[TRACE] ClipComparison.run - RETURNING error result", file=sys.stderr, flush=True)
            meta = {
                "DEBUG_STAGE": "run",
                "DEBUG_ERROR_TYPE": type(exc).__name__,
                "DEBUG_ERROR_MESSAGE": str(exc),
                "DEBUG_TRACEBACK_LINES": tb.splitlines(),
            }
            return build_clip_comparison_response(context=self, similarities={}, meta=meta)


if "__main__" == __name__:
    Executor(sys.argv[1]).run()
