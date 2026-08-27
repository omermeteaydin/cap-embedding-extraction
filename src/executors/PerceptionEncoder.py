"""Executor for Perception Encoder-based embedding extraction in NovaVision pipeline.

GPU/CUDA is required. Not verified end-to-end in this environment because
the `perception_models` package is not installed -- see the src/utils/utils.py
docstring and DOCUMENTATION.md section 8.2.
"""

import os
import sys
import traceback

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.base.capsule import Capsule
from sdks.novavision.src.helper.executor import Executor
from sdks.novavision.src.media.image import Image

from capsules.EmbeddingExtraction.src.models.PackageModel import PackageModel
from capsules.EmbeddingExtraction.src.utils.response import build_perception_encoder_response
from capsules.EmbeddingExtraction.src.utils.utils import load_perception_encoder_loader


class PerceptionEncoder(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        input_data = self.request.get_param("inputData")
        if isinstance(input_data, list) and len(input_data) == 1:
            input_data = input_data[0]
        self.input_data = input_data

    @staticmethod
    def bootstrap(config: dict) -> dict:
        # TEMPORARY DEBUG: same reasoning as ClipGenerate.bootstrap -- the
        # platform LOG has been unreliable (only a WebSocket cert error
        # shows up, never the real traceback), so bootstrap failures are
        # caught here and handed to run() via the returned dict instead of
        # raising -- raising here would prevent run() (and outputMeta) from
        # ever being reached.
        # Remove this try/except once platform logging is confirmed reliable.
        try:
            loader = load_perception_encoder_loader(config=config)
            return {"loader": loader, "bootstrap_error": None}
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"[PerceptionEncoder.bootstrap] DEBUG ERROR:\n{tb}", file=sys.stderr, flush=True)
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
        # Remove this try/except once LOG is confirmed reliable again.
        bootstrap_error = self.bootstrap.get("bootstrap_error")
        if bootstrap_error:
            meta = {
                "DEBUG_STAGE": "bootstrap",
                "DEBUG_ERROR_TYPE": bootstrap_error["type"],
                "DEBUG_ERROR_MESSAGE": bootstrap_error["message"],
                "DEBUG_TRACEBACK_LINES": bootstrap_error["traceback"].splitlines(),
            }
            return build_perception_encoder_response(context=self, embedding=[], meta=meta)

        try:
            loader = self.bootstrap["loader"]

            if isinstance(self.input_data, dict) and self.input_data.get("type") == "Image":
                image = Image.get_frame(img=self.input_data, redis_db=self.redis_db)
                image_array = image.value if image is not None else None
                embedding_vector = loader.embed_image(image_array)
                input_type = "image"
            else:
                embedding_vector = loader.embed_text(str(self.input_data))
                input_type = "text"

            meta = {
                "model_family": loader.model_family,
                "model_version": loader.model_version,
                "input_type": input_type,
                "embedding_dim": int(embedding_vector.shape[0]),
            }

            packageModel = build_perception_encoder_response(
                context=self,
                embedding=embedding_vector.tolist(),
                meta=meta,
            )
            return packageModel
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"[PerceptionEncoder.run] DEBUG ERROR:\n{tb}", file=sys.stderr, flush=True)
            meta = {
                "DEBUG_STAGE": "run",
                "DEBUG_ERROR_TYPE": type(exc).__name__,
                "DEBUG_ERROR_MESSAGE": str(exc),
                "DEBUG_TRACEBACK_LINES": tb.splitlines(),
            }
            return build_perception_encoder_response(context=self, embedding=[], meta=meta)


if "__main__" == __name__:
    Executor(sys.argv[1]).run()
