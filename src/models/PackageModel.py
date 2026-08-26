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

    @validator("value", pre=True)
    def unwrap_single_item_list(cls, value):
        if isinstance(value, list):
            if len(value) == 1:
                return value[0]
            raise ValueError(
                f"inputImage.value received a list with {len(value)} items; "
                "expected exactly one image."
            )
        return value

    class Config:
        title = "Image"


class InputComparisonClasses(Input):
    """
    NOT CURRENTLY WIRED (kept for future revert -- see
    ConfigClipComparisonClasses below for the field actually in use).

    Free-text labels to compare the image against, e.g. ["boat", "buoy",
    "obstacle"]. Same role as Roboflow clip_comparison's `classes` field.

    type is "object" (not "list") to match the platform's built-in
    TextInput/List component, which declares its own outputList as type
    "object" (confirmed via the flow's Request/Preview panels: e.g.
    "IJpQxM-outputList": "object"). The flow engine appears to require the
    input's declared type to match the connected output's declared type
    before it will route data between them -- with type="list" here (vs
    "object" upstream) the connection was accepted visually in Screen
    Builder but ClipComparison was never actually invoked at runtime (no
    "Test - Package ... ClipComparison" log line, Outputs stayed empty),
    and deploy failed with "Release not generated. Please check your
    flow." The actual Python value (List[str]) is unchanged -- only this
    routing-metadata label changes.

    Beyond that fix, ClipComparison still never fired even with a fresh
    node and a structurally correct flow. The one thing that reliably
    distinguishes it from the always-working ClipGenerate is that it has
    TWO incoming connections (Image Load + Input Text) instead of one --
    which looks like a separate platform limitation. As a workaround this
    field was moved to a plain Config the user types directly into the
    node (ConfigClipComparisonClasses), eliminating the second connection.
    Revert to this Input once the platform issue is confirmed fixed.
    """
    name: Literal["inputClasses"] = "inputClasses"
    value: List[str] = Field(default_factory=list)
    type: Literal["object"] = "object"

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


class ConfigClipComparisonClasses(Config):
    """
    WORKAROUND field (see the long comment on InputComparisonClasses
    above for why): the class labels to compare the image against, typed
    directly into the node instead of wired in from a separate Input Text
    component. Comma-separated, e.g. "boat,buoy,person,building".

    Deliberately a plain TOP-LEVEL field on ClipComparisonConfigs (a
    sibling of configClipComparisonAdvance) rather than nested inside the
    Advance/Enable dependent-dropdown -- nested "Advance > Enable"
    sub-fields (like configClipComparisonVersion normally is) were
    confirmed NOT to render at all in the current platform UI, for every
    executor in this package, not just this one. Top-level fields (Task,
    Advance itself) render fine, so this stays top-level until that
    separate platform bug is fixed.
    """
    name: Literal["clipComparisonClasses"] = "ClipComparisonClasses"
    value: str = ""
    type: Literal["string"] = "string"
    field: Literal["widget"] = "widget"

    class Config:
        title = "Classes"
        json_schema_extra = {
            "shortDescription": "Comma-separated labels"
        }


class ClipComparisonConfigs(Configs):
    configClipComparisonAdvance: ConfigClipComparisonAdvance
    configClipComparisonClasses: ConfigClipComparisonClasses


class ClipComparisonInputs(Inputs):
    inputImage: InputComparisonImage


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