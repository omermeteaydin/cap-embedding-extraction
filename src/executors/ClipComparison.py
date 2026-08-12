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
        self.input_classes = self.request.get_param("inputClasses")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        loader = load_clip_comparison_loader(config=config)
        return {"loader": loader}

    def run(self):
        loader = self.bootstrap["loader"]

        image = Image.get_frame(img=self.input_image, redis_db=self.redis_db)
        image_array = image.value if image is not None else None
        classes = list(self.input_classes) if self.input_classes else []

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


if "__main__" == __name__:
    Executor(sys.argv[1]).run()
