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
    ClipGenerateConfigs, ClipGenerateInputs, ClipGenerateRequest, ClipGenerateExecutor,
    ConfigClipGenerateVersion, ConfigClipGenerateNormalize,
    ConfigClipGenerateAdvance, ConfigClipGenerateAdvanceTrue,
    InputComparisonImage, InputComparisonClasses,
    ClipComparisonConfigs, ClipComparisonInputs, ClipComparisonRequest, ClipComparisonExecutor,
    ConfigClipComparisonVersion, ConfigClipComparisonAdvance, ConfigClipComparisonAdvanceTrue,
)

ENDPOINT_URL = "http://127.0.0.1:8000/api"


def _build_clip_generate_configs():
    """Carries version/normalize settings under ConfigClipGenerateAdvance=True --
    same pattern as the ConfigBoTSortAdvance usage in `ObjectTracking`."""
    configClipGenerateVersion = ConfigClipGenerateVersion(value="ViT-B-16")
    configClipGenerateNormalize = ConfigClipGenerateNormalize(value=True)
    configClipGenerateAdvanceTrue = ConfigClipGenerateAdvanceTrue(
        configClipGenerateVersion=configClipGenerateVersion,
        configClipGenerateNormalize=configClipGenerateNormalize,
    )
    configClipGenerateAdvance = ConfigClipGenerateAdvance(value=configClipGenerateAdvanceTrue)
    return ClipGenerateConfigs(configClipGenerateAdvance=configClipGenerateAdvance)


def _build_clip_comparison_configs():
    configClipComparisonVersion = ConfigClipComparisonVersion(value="ViT-B-16")
    configClipComparisonAdvanceTrue = ConfigClipComparisonAdvanceTrue(
        configClipComparisonVersion=configClipComparisonVersion,
    )
    configClipComparisonAdvance = ConfigClipComparisonAdvance(value=configClipComparisonAdvanceTrue)
    return ClipComparisonConfigs(configClipComparisonAdvance=configClipComparisonAdvance)


def inference_generate_image():
    """Example: producing a CLIP embedding from an image (ClipGenerate)."""
    imread = cv2.imread("/opt/project/capsules/EmbeddingExtraction/resources/sample.jpg")
    image_obj = Image(name="Sample", uID="001", mimeType="image/jpg", encoding="bytes", value=imread, type="Image")
    image_obj = image.encode64(image_obj)
    inputData = InputData(value=image_obj)

    clipConfigs = _build_clip_generate_configs()
    clipInputs = ClipGenerateInputs(inputData=inputData)
    clipRequest = ClipGenerateRequest(inputs=clipInputs, configs=clipConfigs)
    clipExecutor = ClipGenerateExecutor(value=clipRequest)
    executor = ConfigExecutor(value=clipExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    request = PackageModel(configs=packageConfigs, name="EmbeddingExtraction", mode="continuous")

    request_json = json.loads(request.json())
    response = requests.post(ENDPOINT_URL, json=request_json)
    print(response.raise_for_status())
    print(response.json())


def inference_generate_text():
    """Example: producing a CLIP embedding from free text (ClipGenerate)."""
    inputData = InputData(value="a red buoy on water")

    clipConfigs = _build_clip_generate_configs()
    clipInputs = ClipGenerateInputs(inputData=inputData)
    clipRequest = ClipGenerateRequest(inputs=clipInputs, configs=clipConfigs)
    clipExecutor = ClipGenerateExecutor(value=clipRequest)
    executor = ConfigExecutor(value=clipExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    request = PackageModel(configs=packageConfigs, name="EmbeddingExtraction", mode="continuous")

    request_json = json.loads(request.json())
    response = requests.post(ENDPOINT_URL, json=request_json)
    print(response.raise_for_status())
    print(response.json())


def inference_comparison():
    """Example: zero-shot classification of an image against text labels
    (ClipComparison)."""
    imread = cv2.imread("/opt/project/capsules/EmbeddingExtraction/resources/sample.jpg")
    image_obj = Image(name="Sample", uID="001", mimeType="image/jpg", encoding="bytes", value=imread, type="Image")
    image_obj = image.encode64(image_obj)
    inputImage = InputComparisonImage(value=image_obj)
    inputClasses = InputComparisonClasses(value=["boat", "buoy", "obstacle"])

    comparisonConfigs = _build_clip_comparison_configs()
    comparisonInputs = ClipComparisonInputs(inputImage=inputImage, inputClasses=inputClasses)
    comparisonRequest = ClipComparisonRequest(inputs=comparisonInputs, configs=comparisonConfigs)
    comparisonExecutor = ClipComparisonExecutor(value=comparisonRequest)
    executor = ConfigExecutor(value=comparisonExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    request = PackageModel(configs=packageConfigs, name="EmbeddingExtraction", mode="continuous")

    request_json = json.loads(request.json())
    response = requests.post(ENDPOINT_URL, json=request_json)
    print(response.raise_for_status())
    print(response.json())


if __name__ == "__main__":
    inference_generate_image()
    # inference_generate_text()
    # inference_comparison()
