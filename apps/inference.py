import os
import sys
import requests
import cv2
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from sdks.novavision.src.media.image import Image as image
from sdks.novavision.src.base.model import Image
from capsules.EmbeddingExtraction.src.models.PackageModel import (
    InputData, PackageConfigs, ConfigExecutor, PackageModel,
    ClipEmbeddingConfigs, ClipEmbeddingInputs, ClipEmbeddingRequest, ClipEmbeddingExecutor,
    ConfigClipVersion, ConfigClipNormalize, ConfigClipAdvance, ConfigClipAdvanceTrue,
)

ENDPOINT_URL = "http://127.0.0.1:8000/api"


def _build_clip_configs():
    """Carries version/normalize settings under ConfigClipAdvance=True --
    same pattern as the ConfigBoTSortAdvance usage in `ObjectTracking`."""
    configClipVersion = ConfigClipVersion(value="ViT-B-16")
    configClipNormalize = ConfigClipNormalize(value=True)
    configClipAdvanceTrue = ConfigClipAdvanceTrue(
        configClipVersion=configClipVersion,
        configClipNormalize=configClipNormalize,
    )
    configClipAdvance = ConfigClipAdvance(value=configClipAdvanceTrue)
    return ClipEmbeddingConfigs(configClipAdvance=configClipAdvance)


def inference_image():
    """Example: producing a CLIP embedding from an image."""
    imread = cv2.imread("/opt/project/capsules/EmbeddingExtraction/resources/sample.jpg")
    image_obj = Image(name="Sample", uID="001", mimeType="image/jpg", encoding="bytes", value=imread, type="Image")
    image_obj = image.encode64(image_obj)
    inputData = InputData(value=image_obj)

    clipConfigs = _build_clip_configs()
    clipInputs = ClipEmbeddingInputs(inputData=inputData)
    clipRequest = ClipEmbeddingRequest(inputs=clipInputs, configs=clipConfigs)
    clipExecutor = ClipEmbeddingExecutor(value=clipRequest)
    executor = ConfigExecutor(value=clipExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    request = PackageModel(configs=packageConfigs, name="EmbeddingExtraction", mode="continuous")

    request_json = json.loads(request.json())
    response = requests.post(ENDPOINT_URL, json=request_json)
    print(response.raise_for_status())
    print(response.json())


def inference_text():
    """Example: producing a CLIP embedding from free text."""
    inputData = InputData(value="a red buoy on water")

    clipConfigs = _build_clip_configs()
    clipInputs = ClipEmbeddingInputs(inputData=inputData)
    clipRequest = ClipEmbeddingRequest(inputs=clipInputs, configs=clipConfigs)
    clipExecutor = ClipEmbeddingExecutor(value=clipRequest)
    executor = ConfigExecutor(value=clipExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    request = PackageModel(configs=packageConfigs, name="EmbeddingExtraction", mode="continuous")

    request_json = json.loads(request.json())
    response = requests.post(ENDPOINT_URL, json=request_json)
    print(response.raise_for_status())
    print(response.json())


if __name__ == "__main__":
    inference_image()
    # inference_text()
