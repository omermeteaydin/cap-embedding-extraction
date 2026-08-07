from sdks.novavision.src.helper.package import PackageHelper
from capsules.EmbeddingExtraction.src.models.PackageModel import (
    PackageModel,
    PackageConfigs,
    ConfigExecutor,
    ClipEmbeddingResponse,
    ClipEmbeddingExecutor,
    ClipEmbeddingOutputs,
    PerceptionEncoderEmbeddingResponse,
    PerceptionEncoderEmbeddingExecutor,
    PerceptionEncoderEmbeddingOutputs,
    OutputEmbedding,
    OutputMeta,
)


def build_clip_response(context, embedding, meta):
    outputEmbedding = OutputEmbedding(value=embedding)
    outputMeta = OutputMeta(value=meta)
    clipOutputs = ClipEmbeddingOutputs(outputEmbedding=outputEmbedding, outputMeta=outputMeta)
    clipResponse = ClipEmbeddingResponse(outputs=clipOutputs)
    clipExecutor = ClipEmbeddingExecutor(value=clipResponse)

    executor = ConfigExecutor(value=clipExecutor)
    packageConfigs = PackageConfigs(executor=executor)

    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel


def build_perception_encoder_response(context, embedding, meta):
    outputEmbedding = OutputEmbedding(value=embedding)
    outputMeta = OutputMeta(value=meta)
    peOutputs = PerceptionEncoderEmbeddingOutputs(
        outputEmbedding=outputEmbedding, outputMeta=outputMeta
    )
    peResponse = PerceptionEncoderEmbeddingResponse(outputs=peOutputs)
    peExecutor = PerceptionEncoderEmbeddingExecutor(value=peResponse)

    executor = ConfigExecutor(value=peExecutor)
    packageConfigs = PackageConfigs(executor=executor)

    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel
