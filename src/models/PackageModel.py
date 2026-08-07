from pydantic import validator
from typing import List, Optional, Union, Literal

from sdks.novavision.src.base.model import Package, Configs, Outputs, Inputs, \
    Response, Request, Output, Input, Config, Image


class InputData(Input):
    """
    CLIP / Perception Encoder ortak girdisi: tek bir görüntü VEYA serbest metin.
    Roboflow'un `data` (Union[image, string]) sözleşmesiyle aynı mantık.
    """
    name: Literal["inputData"] = "inputData"
    value: Union[Image, str]
    type: str = "object"

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
    Embedding'in hangi model ailesi/versiyonuyla üretildiğine dair iz bilgisi.
    Farklı model versiyonlarıyla üretilen embedding'lerin yanlışlıkla aynı
    uzaymış gibi kıyaslanmasını önlemek için tutulur.
    """
    name: Literal["outputMeta"] = "outputMeta"
    value: dict
    type: Literal["object"] = "object"

    class Config:
        title = "Meta"


# ==========================================
# 1. CLIP Executor Configurations
# ==========================================


class ConfigClipVersion(Config):
    """CLIP backbone/çözünürlük varyantı."""
    name: Literal["clipVersion"] = "ClipVersion"
    value: Literal["ViT-B-16", "ViT-B-32", "RN50"] = "ViT-B-16"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "CLIP Version"
        json_schema_extra = {
            "shortDescription": "Model Variant"
        }


class ConfigClipNormalize(Config):
    """Çıktı embedding'inin L2-normalize edilip edilmeyeceği."""
    name: Literal["clipNormalize"] = "ClipNormalize"
    value: bool = True
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Normalize Embedding"
        json_schema_extra = {
            "shortDescription": "L2 Normalize"
        }


class ConfigClipAdvanceTrue(Config):
    name: Literal["True"] = "True"
    value: Literal["True"] = "True"
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"
    configClipVersion: ConfigClipVersion
    configClipNormalize: ConfigClipNormalize

    class Config:
        title = "Enable"


class ConfigClipAdvanceFalse(Config):
    name: Literal["False"] = "False"
    value: Literal["False"] = "False"
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Disable"


class ConfigClipAdvance(Config):
    """
    Enable advanced settings for CLIP embedding (version, normalize).
    `ObjectTracking`'deki ConfigBoTSortAdvance ile aynı desen: model
    yükleme kararını etkileyen config'ler bootstrap()'ı yeniden tetiklemesi
    gerektiği için restart=True taşır.
    """
    name: Literal["ConfigClipAdvance"] = "ConfigClipAdvance"
    value: Union[ConfigClipAdvanceTrue, ConfigClipAdvanceFalse]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"
    restart: Literal[True] = True

    class Config:
        title = "Advance"
        json_schema_extra = {
            "shortDescription": "Advanced Settings"
        }


class ClipEmbeddingConfigs(Configs):
    configClipAdvance: ConfigClipAdvance


class ClipEmbeddingInputs(Inputs):
    inputData: InputData


class ClipEmbeddingOutputs(Outputs):
    outputEmbedding: OutputEmbedding
    outputMeta: OutputMeta


class ClipEmbeddingRequest(Request):
    inputs: Optional[ClipEmbeddingInputs] = None
    configs: ClipEmbeddingConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class ClipEmbeddingResponse(Response):
    outputs: ClipEmbeddingOutputs


class ClipEmbeddingExecutor(Config):
    name: Literal["ClipEmbedding"] = "ClipEmbedding"
    value: Union[ClipEmbeddingRequest, ClipEmbeddingResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "ClipEmbedding"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


# ==========================================
# 2. Perception Encoder Executor Configurations
# ==========================================
# ⚠ GPU/CUDA zorunludur (Roboflow'un roboflow_core/perception_encoder@v1
#   bloğu da aynı kısıtı belgeliyor). Bu ortamda `perception_models`
#   paketi kurulu olmadığından uçtan uca doğrulanamamıştır.


class ConfigPerceptionEncoderVersion(Config):
    """Perception Encoder backbone/çözünürlük varyantı."""
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


class PerceptionEncoderEmbeddingConfigs(Configs):
    configPerceptionEncoderAdvance: ConfigPerceptionEncoderAdvance


class PerceptionEncoderEmbeddingInputs(Inputs):
    inputData: InputData


class PerceptionEncoderEmbeddingOutputs(Outputs):
    outputEmbedding: OutputEmbedding
    outputMeta: OutputMeta


class PerceptionEncoderEmbeddingRequest(Request):
    inputs: Optional[PerceptionEncoderEmbeddingInputs] = None
    configs: PerceptionEncoderEmbeddingConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class PerceptionEncoderEmbeddingResponse(Response):
    outputs: PerceptionEncoderEmbeddingOutputs


class PerceptionEncoderEmbeddingExecutor(Config):
    name: Literal["PerceptionEncoderEmbedding"] = "PerceptionEncoderEmbedding"
    value: Union[PerceptionEncoderEmbeddingRequest, PerceptionEncoderEmbeddingResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "PerceptionEncoderEmbedding"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


# ==========================================
# 3. Global Package Configuration
# ==========================================


class ConfigExecutor(Config):
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[
        ClipEmbeddingExecutor,
        PerceptionEncoderEmbeddingExecutor,
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
