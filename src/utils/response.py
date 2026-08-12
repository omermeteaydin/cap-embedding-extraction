from sdks.novavision.src.helper.package import PackageHelper
from capsules.EmbeddingExtraction.src.models.PackageModel import (
    PackageModel,
    PackageConfigs,
    ConfigExecutor,
    ClipGenerateResponse,
    ClipGenerateExecutor,
    ClipGenerateOutputs,
    ClipComparisonResponse,
    ClipComparisonExecutor,
    ClipComparisonOutputs,
    PerceptionEncoderResponse,
    PerceptionEncoderExecutor,
    PerceptionEncoderOutputs,
    OutputEmbedding,
    OutputMeta,
    OutputSimilarities,
)


def build_clip_generate_response(context, embedding, meta):
    outputEmbedding = OutputEmbedding(value=embedding)
    outputMeta = OutputMeta(value=meta)
    clipOutputs = ClipGenerateOutputs(outputEmbedding=outputEmbedding, outputMeta=outputMeta)
    clipResponse = ClipGenerateResponse(outputs=clipOutputs)
    clipExecutor = ClipGenerateExecutor(value=clipResponse)

    executor = ConfigExecutor(value=clipExecutor)
    packageConfigs = PackageConfigs(executor=executor)

    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel


def build_clip_comparison_response(context, similarities, meta):
    outputSimilarities = OutputSimilarities(value=similarities)
    outputMeta = OutputMeta(value=meta)
    comparisonOutputs = ClipComparisonOutputs(
        outputSimilarities=outputSimilarities, outputMeta=outputMeta
    )
    comparisonResponse = ClipComparisonResponse(outputs=comparisonOutputs)
    comparisonExecutor = ClipComparisonExecutor(value=comparisonResponse)

    executor = ConfigExecutor(value=comparisonExecutor)
    packageConfigs = PackageConfigs(executor=executor)

    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel


def build_perception_encoder_response(context, embedding, meta):
    outputEmbedding = OutputEmbedding(value=embedding)
    outputMeta = OutputMeta(value=meta)
    peOutputs = PerceptionEncoderOutputs(
        outputEmbedding=outputEmbedding, outputMeta=outputMeta
    )
    peResponse = PerceptionEncoderResponse(outputs=peOutputs)
    peExecutor = PerceptionEncoderExecutor(value=peResponse)

    executor = ConfigExecutor(value=peExecutor)
    packageConfigs = PackageConfigs(executor=executor)

    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel
