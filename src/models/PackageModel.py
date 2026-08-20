from pydantic import validator, Field
from typing import Dict, List, Optional, Union, Literal

from sdks.novavision.src.base.model import Package, Configs, Outputs, Inputs, \
    Response, Request, Output, Input, Config, Image


class InputData(Input):
    """
    Shared input for embedding generation: a single image OR free text.
    Same contract as Roboflow's `data` field (Union[image, string]).
    """
    name: Literal["inputData"] = "inputData"
    value: Union[Image, str]
    type: str = "object"

    @validator("value", pre=True)
    def unwrap_single_item_list(cls, value):
        """
        The platform's ImageLoad component sometimes sends the image
        wrapped in a single-item list (e.g. [{"name": "Image_...", ...}])
        instead of a bare Image dict. If we receive a list with exactly
        one item, unwrap it so downstream validation (Union[Image, str])
        succeeds as before. A list with more/fewer than one item is
        rejected explicitly rather than silently guessing.
        """
        if isinstance(value, list):
            if len(value) == 1:
                return value[0]
            raise ValueError(
                f"inputData.value received a list with {len(value)} items; "
                "expected exactly one image (batch input is not supported)."
            )
        return value

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get('value')
        if isinstance(value, Image):
            return "object"
        return "string"

    class Config:
        title = "Data"


class OutputEmbedding(Output):
    name: Literal["outputEmbedding"] = "outputEmbedding"
    value: List[float]
    type: Literal["list"] = "list"

    class Config:
        title = "Embedding"


class OutputMeta(Output):
    """
    Traceability info about which model family/version produced the
    output. Kept to prevent accidentally comparing embeddings that
    were produced by different model versions (different vector spaces).
    """
    name: Literal["outputMeta"] = "outputMeta"
    value: dict
    type: Literal["object"] = "object"

    class Config:
        title = "Meta"


# ==========================================
# 1. ClipGenerate Executor Configurations
# ==========================================
# Image or text -> a single embedding vector.
# Mirrors Roboflow's `roboflow_core/clip@v1` block.


class ConfigClipGenerateVersion(Config):
    """CLIP backbone / resolution variant."""
    name: Literal["clipGenerateVersion"] = "ClipGenerateVersion"
    value: Literal["ViT-B-16", "ViT-B-32", "RN50"] = "ViT-B-16"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "CLIP Version"
        json_schema_extra = {
            "shortDescription": "Model Variant"
        }


class ConfigClipGenerateNormalize(Config):
    """Whether the output embedding should be L2-normalized."""
    name: Literal["clipGenerateNormalize"] = "ClipGenerateNormalize"
    value: bool = True
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Normalize Embedding"
        json_schema_extra = {
            "shortDescription": "L2 Normalize"
        }


class ConfigClipGenerateAdvanceTrue(Config):
    name: Literal["True"] = "True"
    value: Literal["True"] = "True"
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"
    configClipGenerateVersion: ConfigClipGenerateVersion
    configClipGenerateNormalize: ConfigClipGenerateNormalize

    class Config:
        title = "Enable"


class ConfigClipGenerateAdvanceFalse(Config):
    name: Literal["False"] = "False"
    value: Literal["False"] = "False"
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Disable"


class ConfigClipGenerateAdvance(Config):
    """
    Enable advanced settings for CLIP embedding generation (version,
    normalize). Same pattern as ConfigBoTSortAdvance in `ObjectTracking`:
    configs that affect the model-loading decision must carry
    restart=True so the platform knows to re-trigger bootstrap().
    """
    name: Literal["ConfigClipGenerateAdvance"] = "ConfigClipGenerateAdvance"
    value: Union[ConfigClipGenerateAdvanceTrue, ConfigClipGenerateAdvanceFalse]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"
    restart: Literal[True] = True

    class Config:
        title = "Advance"
        json_schema_extra = {
            "shortDescription": "Advanced Settings"
        }


class ClipGenerateConfigs(Configs):
    configClipGenerateAdvance: ConfigClipGenerateAdvance


class ClipGenerateInputs(Inputs):
    inputData: InputData


class ClipGenerateOutputs(Outputs):
    outputEmbedding: OutputEmbedding
    outputMeta: OutputMeta


class ClipGenerateRequest(Request):
    inputs: Optional[ClipGenerateInputs] = None
    configs: ClipGenerateConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class ClipGenerateResponse(Response):
    outputs: ClipGenerateOutputs


class ClipGenerateExecutor(Config):
    name: Literal["ClipGenerate"] = "ClipGenerate"
    value: Union[ClipGenerateRequest, ClipGenerateResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Clip (Generate)"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


# ==========================================
# 2. ClipComparison Executor Configurations
# ==========================================
# Image + a list of text labels -> similarity score per label.
# Mirrors Roboflow's `roboflow_core/clip_comparison@v2` block
# (zero-shot classification without training a dedicated model).


class InputComparisonImage(Input):
    """The image to classify against the provided text labels."""
    name: Literal["inputImage"] = "inputImage"
    value: Image
    type: Literal["object"] = "object"

    class Config:
        title = "Image"


class InputComparisonClasses(Input):
    """
    Free-text labels to compare the image against, e.g. ["boat", "buoy",
    "obstacle"]. Same role as Roboflow clip_comparison's `classes` field.
    """
    name: Literal["inputClasses"] = "inputClasses"
    value: List[str] = Field(default_factory=list)
    type: Literal["list"] = "list"

    class Config:
        title = "Classes"


class ConfigClipComparisonVersion(Config):
    """CLIP backbone / resolution variant."""
    name: Literal["clipComparisonVersion"] = "ClipComparisonVersion"
    value: Literal["ViT-B-16", "ViT-B-32", "RN50"] = "ViT-B-16"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "CLIP Version"
        json_schema_extra = {
            "shortDescription": "Model Variant"
        }


class ConfigClipComparisonAdvanceTrue(Config):
    name: Literal["True"] = "True"
    value: Literal["True"] = "True"
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"
    configClipComparisonVersion: ConfigClipComparisonVersion

    class Config:
        title = "Enable"


class ConfigClipComparisonAdvanceFalse(Config):
    name: Literal["False"] = "False"
    value: Literal["False"] = "False"
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Disable"


class ConfigClipComparisonAdvance(Config):
    """Enable advanced settings for CLIP comparison (version selection)."""
    name: Literal["ConfigClipComparisonAdvance"] = "ConfigClipComparisonAdvance"
    value: Union[ConfigClipComparisonAdvanceTrue, ConfigClipComparisonAdvanceFalse]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"
    restart: Literal[True] = True

    class Config:
        title = "Advance"
        json_schema_extra = {
            "shortDescription": "Advanced Settings"
        }


class ClipComparisonConfigs(Configs):
    configClipComparisonAdvance: ConfigClipComparisonAdvance


class ClipComparisonInputs(Inputs):
    inputImage: InputComparisonImage
    inputClasses: InputComparisonClasses


class OutputSimilarities(Output):
    """
    Similarity score (0-1, cosine similarity rescaled) for each provided
    class label, e.g. {"boat": 0.82, "buoy": 0.41, "obstacle": 0.09}.
    """
    name: Literal["outputSimilarities"] = "outputSimilarities"
    value: Dict[str, float]
    type: Literal["object"] = "object"

    class Config:
        title = "Similarities"


class ClipComparisonOutputs(Outputs):
    outputSimilarities: OutputSimilarities
    outputMeta: OutputMeta


class ClipComparisonRequest(Request):
    inputs: Optional[ClipComparisonInputs] = None
    configs: ClipComparisonConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class ClipComparisonResponse(Response):
    outputs: ClipComparisonOutputs


class ClipComparisonExecutor(Config):
    name: Literal["ClipComparison"] = "ClipComparison"
    value: Union[ClipComparisonRequest, ClipComparisonResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Clip (Comparison)"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


# ==========================================
# 3. PerceptionEncoder Executor Configurations
# ==========================================
# Image or text -> a single embedding vector (Generate only; Roboflow
# does not currently offer a dedicated Perception Encoder comparison
# block, so this package doesn't either).
# GPU/CUDA is required (Roboflow's own roboflow_core/perception_encoder@v1
# block documents the same constraint). Not verified end-to-end in this
# environment because the `perception_models` package is not installed.


class ConfigPerceptionEncoderVersion(Config):
    """Perception Encoder backbone / resolution variant."""
    name: Literal["perceptionEncoderVersion"] = "PerceptionEncoderVersion"
    value: Literal["PE-Core-B16-224", "PE-Core-L14-336"] = "PE-Core-B16-224"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Perception Encoder Version"
        json_schema_extra = {
            "shortDescription": "Model Variant"
        }


class ConfigPerceptionEncoderNormalize(Config):
    name: Literal["perceptionEncoderNormalize"] = "PerceptionEncoderNormalize"
    value: bool = True
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Normalize Embedding"
        json_schema_extra = {
            "shortDescription": "L2 Normalize"
        }


class ConfigPerceptionEncoderAdvanceTrue(Config):
    name: Literal["True"] = "True"
    value: Literal["True"] = "True"
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"
    configPerceptionEncoderVersion: ConfigPerceptionEncoderVersion
    configPerceptionEncoderNormalize: ConfigPerceptionEncoderNormalize

    class Config:
        title = "Enable"


class ConfigPerceptionEncoderAdvanceFalse(Config):
    name: Literal["False"] = "False"
    value: Literal["False"] = "False"
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Disable"


class ConfigPerceptionEncoderAdvance(Config):
    """Enable advanced settings for Perception Encoder embedding."""
    name: Literal["ConfigPerceptionEncoderAdvance"] = "ConfigPerceptionEncoderAdvance"
    value: Union[ConfigPerceptionEncoderAdvanceTrue, ConfigPerceptionEncoderAdvanceFalse]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"
    restart: Literal[True] = True

    class Config:
        title = "Advance"
        json_schema_extra = {
            "shortDescription": "Advanced Settings"
        }


class PerceptionEncoderConfigs(Configs):
    configPerceptionEncoderAdvance: ConfigPerceptionEncoderAdvance


class PerceptionEncoderInputs(Inputs):
    inputData: InputData


class PerceptionEncoderOutputs(Outputs):
    outputEmbedding: OutputEmbedding
    outputMeta: OutputMeta


class PerceptionEncoderRequest(Request):
    inputs: Optional[PerceptionEncoderInputs] = None
    configs: PerceptionEncoderConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class PerceptionEncoderResponse(Response):
    outputs: PerceptionEncoderOutputs


class PerceptionEncoderExecutor(Config):
    name: Literal["PerceptionEncoder"] = "PerceptionEncoder"
    value: Union[PerceptionEncoderRequest, PerceptionEncoderResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Perception Encoder"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


# ==========================================
# 4. Global Package Configuration
# ==========================================


class ConfigExecutor(Config):
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[
        ClipGenerateExecutor,
        ClipComparisonExecutor,
        PerceptionEncoderExecutor,
    ]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Task"


class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    configs: PackageConfigs
    type: Literal["capsule"] = "capsule"
    name: Literal["EmbeddingExtraction"] = "EmbeddingExtraction"
