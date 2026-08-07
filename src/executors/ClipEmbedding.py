"""Executor for CLIP-based embedding extraction in NovaVision pipeline.

Handles single image/text ingestion and semantic embedding generation
using an open_clip backbone (ViT-B-16, ViT-B-32 or RN50).
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.base.capsule import Capsule
from sdks.novavision.src.helper.executor import Executor
from sdks.novavision.src.media.image import Image

from capsules.EmbeddingExtraction.src.models.PackageModel import PackageModel
from capsules.EmbeddingExtraction.src.utils.response import build_clip_response
from capsules.EmbeddingExtraction.src.utils.utils import load_clip_loader


class ClipEmbedding(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.input_data = self.request.get_param("inputData")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        loader = load_clip_loader(config=config)
        return {"loader": loader}

    def run(self):
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

        packageModel = build_clip_response(
            context=self,
            embedding=embedding_vector.tolist(),
            meta=meta,
        )
        return packageModel


if "__main__" == __name__:
    Executor(sys.argv[1]).run()
