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

from capsules.EmbeddingExtraction.src.models.PackageModel import PackageModel
from capsules.EmbeddingExtraction.src.utils.response import build_clip_comparison_response
from capsules.EmbeddingExtraction.src.utils.utils import load_clip_comparison_loader


class ClipComparison(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.input_image = self.request.get_param("inputImage")
        # WORKAROUND: Classes used to be a wired Input (inputClasses, see
        # InputComparisonClasses in PackageModel.py for why it was moved).
        # It is now a plain Config the user types directly into the node:
        # a comma-separated string, e.g. "boat,buoy,person,building".
        self.classes_raw = self.request.get_param("clipComparisonClasses")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        # TEMPORARY DEBUG: same reasoning as ClipGenerate.bootstrap -- the
        # platform LOG only shows a bare exception message (no traceback,
        # e.g. just "'value'" for a KeyError), so bootstrap failures are
        # caught here and handed to run() via the returned dict instead of
        # raising -- raising here would prevent run() (and outputMeta) from
        # ever being reached.
        # Remove this try/except once platform logging is confirmed reliable.
        try:
            loader = load_clip_comparison_loader(config=config)
            return {"loader": loader, "bootstrap_error": None}
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"[ClipComparison.bootstrap] DEBUG ERROR:\n{tb}", file=sys.stderr, flush=True)
            return {
                "loader": None,
                "bootstrap_error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": tb,
                },
            }

    def run(self):
        # TEMPORARY DEBUG: same reasoning as ClipGenerate.run -- surfaces the
        # real exception (type + message + traceback) inside outputMeta so it
        # is visible in the flow's Output/Raw tab without depending on
        # platform LOG streaming. The traceback is split into a list of
        # lines (rather than one long string) because the platform's UI
        # misinterpreted a long single string as base64 image data and
        # rendered an "Image Preview" instead of the text.
        # Remove this try/except once platform LOG is confirmed reliable.
        bootstrap_error = self.bootstrap.get("bootstrap_error")
        if bootstrap_error:
            meta = {
                "DEBUG_STAGE": "bootstrap",
                "DEBUG_ERROR_TYPE": bootstrap_error["type"],
                "DEBUG_ERROR_MESSAGE": bootstrap_error["message"],
                "DEBUG_TRACEBACK_LINES": bootstrap_error["traceback"].splitlines(),
            }
            return build_clip_comparison_response(context=self, similarities={}, meta=meta)

        try:
            loader = self.bootstrap["loader"]

            image = Image.get_frame(img=self.input_image, redis_db=self.redis_db)
            image_array = image.value if image is not None else None
            classes = [
                label.strip()
                for label in (self.classes_raw or "").split(",")
                if label.strip()
            ]

            similarities = loader.compare_image_to_texts(image_array, classes)

            meta = {
                "model_family": loader.model_family,
                "model_version": loader.model_version,
                "num_classes": len(classes),
            }

            packageModel = build_clip_comparison_response(
                context=self,
                similarities=similarities,
                meta=meta,
            )
            return packageModel
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"[ClipComparison.run] DEBUG ERROR:\n{tb}", file=sys.stderr, flush=True)
            meta = {
                "DEBUG_STAGE": "run",
                "DEBUG_ERROR_TYPE": type(exc).__name__,
                "DEBUG_ERROR_MESSAGE": str(exc),
                "DEBUG_TRACEBACK_LINES": tb.splitlines(),
            }
            return build_clip_comparison_response(context=self, similarities={}, meta=meta)


if "__main__" == __name__:
    Executor(sys.argv[1]).run()