"""
utils.py

CLIP / Perception Encoder model loading, embedding extraction, and
comparison helpers. Written in the same spirit as the appearance-extractor
pattern in `BoTSORTTracker/reid.py` (cropping, batching, normalization,
caching).

Notes:
- The CLIP path is fully functional via `open_clip_torch`
  (pip install open_clip_torch); GPU is not required, it also runs on CPU.
- The Perception Encoder path depends on Meta's `perception_models` package
  and requires GPU/CUDA (see Roboflow's roboflow_core/perception_encoder@v1
  block documentation). Written as best-effort since the package is not
  installed in this environment.
"""

from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import torch
from PIL import Image as PILImage

from sdks.novavision.src.base.application import Application


CLIP_VERSION_TO_OPEN_CLIP = {
    # version string -> (open_clip model_name, pretrained tag)
    "ViT-B-16": ("ViT-B-16", "openai"),
    "ViT-B-32": ("ViT-B-32", "openai"),
    "RN50": ("RN50", "openai"),
}

PERCEPTION_ENCODER_VERSIONS = {"PE-Core-B16-224", "PE-Core-L14-336"}


def _build_clip_generate_cfg(config):
    """Same pattern as `_build_bot_sort_cfg`: reads the
    ConfigClipGenerateAdvance toggle via Application().get_param(); if
    True, reads the sub-configs, if False, uses defaults. Config values
    are read ONLY here, inside bootstrap() -- request.get_param() is
    used only for inputs."""
    application = Application()
    advance = application.get_param(config=config, name="ConfigClipGenerateAdvance")

    if advance == "True":
        version = application.get_param(config=config, name="ClipGenerateVersion")
        normalize = application.get_param(config=config, name="ClipGenerateNormalize")
        return SimpleNamespace(
            model_family="CLIP",
            model_version=version or "ViT-B-16",
            normalize=normalize if normalize is not None else True,
            device="cpu",  # GPU is not required for CLIP; CUDA is used automatically if available
        )

    return SimpleNamespace(
        model_family="CLIP",
        model_version="ViT-B-16",
        normalize=True,
        device="cpu",
    )


def _build_clip_comparison_cfg(config):
    """Same pattern as `_build_clip_generate_cfg`, but for the
    ClipComparison executor (no `normalize` toggle -- similarity scores
    are always computed on normalized embeddings, see
    EmbeddingModelLoader.compare_image_to_texts)."""
    application = Application()
    advance = application.get_param(config=config, name="ConfigClipComparisonAdvance")

    if advance == "True":
        version = application.get_param(config=config, name="ClipComparisonVersion")
        return SimpleNamespace(
            model_family="CLIP",
            model_version=version or "ViT-B-16",
            normalize=True,
            device="cpu",
        )

    return SimpleNamespace(
        model_family="CLIP",
        model_version="ViT-B-16",
        normalize=True,
        device="cpu",
    )


def _build_perception_encoder_cfg(config):
    application = Application()
    advance = application.get_param(config=config, name="ConfigPerceptionEncoderAdvance")

    if advance == "True":
        version = application.get_param(config=config, name="PerceptionEncoderVersion")
        normalize = application.get_param(config=config, name="PerceptionEncoderNormalize")
        return SimpleNamespace(
            model_family="PerceptionEncoder",
            model_version=version or "PE-Core-B16-224",
            normalize=normalize if normalize is not None else True,
            device="cuda",  # Perception Encoder requires GPU/CUDA
        )

    return SimpleNamespace(
        model_family="PerceptionEncoder",
        model_version="PE-Core-B16-224",
        normalize=True,
        device="cuda",
    )


def resolve_model_version(model_family: str, model_version: str) -> str:
    if model_family == "CLIP":
        if model_version not in CLIP_VERSION_TO_OPEN_CLIP:
            raise ValueError(
                f"Unknown CLIP version: '{model_version}'. "
                f"Supported: {list(CLIP_VERSION_TO_OPEN_CLIP.keys())}"
            )
    elif model_family == "PerceptionEncoder":
        if model_version not in PERCEPTION_ENCODER_VERSIONS:
            raise ValueError(
                f"Unknown Perception Encoder version: '{model_version}'. "
                f"Supported: {sorted(PERCEPTION_ENCODER_VERSIONS)}"
            )
    else:
        raise ValueError(f"Unknown model family: '{model_family}'")
    return model_version


class EmbeddingModelLoader:
    """Loads and caches the CLIP or Perception Encoder model.
    Created once inside `bootstrap()`, reused across `run()` calls
    (same lifecycle as the ReID class in reid.py).

    `normalize` is read from config at bootstrap-time and fixed here --
    it is not re-read from config at runtime (inside `run()`), because
    the reference SDK pattern only reads configs inside bootstrap() via
    Application().get_param() (see utils.py:_build_clip_generate_cfg)."""

    def __init__(self, model_family: str, model_version: str, normalize: bool = True,
                 device: str = None):
        self.model_family = model_family
        self.model_version = resolve_model_version(model_family, model_version)
        self.normalize = normalize
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self._load()

    def _load(self):
        if self.model_family == "CLIP":
            self._load_clip()
        elif self.model_family == "PerceptionEncoder":
            self._load_perception_encoder()
        else:
            raise ValueError(f"Unknown model family: '{self.model_family}'")

    def _load_clip(self):
        import open_clip  # pip install open_clip_torch

        model_name, pretrained = CLIP_VERSION_TO_OPEN_CLIP[self.model_version]
        # OpenAI's original CLIP weights (ViT-B-16/32, RN50) were trained
        # with QuickGELU activation (open_clip.get_pretrained_cfg(...)["quick_gelu"]
        # == True). Without force_quick_gelu, the model is built with the
        # default (standard) GELU and the weights end up loaded with the
        # wrong activation function -- the forward pass still runs, but
        # embeddings drift from what the model produced at training time.
        # Verified: force_quick_gelu=False -> nn.GELU, True -> nn.QuickGELU
        # (tested in this environment via open_clip.get_pretrained_cfg and
        # by inspecting the actual layer type).
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, force_quick_gelu=True
        )
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def _load_perception_encoder(self):
        """Not verified in this environment -- see module docstring."""
        try:
            import core.vision_encoder.pe as pe  # type: ignore
            import core.vision_encoder.transforms as pe_transforms  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Perception Encoder package not found. Install with: "
                "pip install git+https://github.com/facebookresearch/perception_models.git "
                "-- GPU/CUDA is required."
            ) from exc

        model = pe.CLIP.from_config(self.model_version, pretrained=True)
        self.model = model.to(self.device).eval()
        self.preprocess = pe_transforms.get_image_transform(model.image_size)
        self.tokenizer = pe_transforms.get_text_tokenizer(model.context_length)

    @torch.no_grad()
    def embed_image(self, image_array: np.ndarray, normalize: bool = None) -> np.ndarray:
        """NumPy (H, W, 3) RGB array -> embedding vector.
        If `normalize` is not given (None), the bootstrap-time
        `self.normalize` read from config is used."""
        pil_image = PILImage.fromarray(image_array.astype("uint8"), "RGB")
        tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        features = self.model.encode_image(tensor)
        return self._postprocess(features, self.normalize if normalize is None else normalize)

    @torch.no_grad()
    def embed_text(self, text: str, normalize: bool = None) -> np.ndarray:
        """Free text -> embedding vector."""
        tokens = self.tokenizer([text]).to(self.device)
        features = self.model.encode_text(tokens)
        return self._postprocess(features, self.normalize if normalize is None else normalize)

    @torch.no_grad()
    def compare_image_to_texts(self, image_array: np.ndarray, texts: List[str]) -> Dict[str, float]:
        """
        Zero-shot classification: embeds `image_array` once, embeds each
        string in `texts`, and returns the cosine similarity between the
        image and each text label. Mirrors Roboflow's
        `roboflow_core/clip_comparison@v2` block (a single image compared
        against a list of class labels).

        Both image and text embeddings are always L2-normalized here
        (regardless of the `self.normalize` bootstrap setting) because
        the cosine-similarity formula requires unit vectors to reduce to
        a plain dot product; this keeps the comparison mathematically
        correct even if a caller instantiates the loader with
        normalize=False for some other purpose.
        """
        if not texts:
            return {}

        pil_image = PILImage.fromarray(image_array.astype("uint8"), "RGB")
        image_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        image_features = self.model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        text_tokens = self.tokenizer(texts).to(self.device)
        text_features = self.model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Cosine similarity == dot product once both sides are unit-normalized.
        similarities = (image_features @ text_features.T).squeeze(0)
        scores = similarities.detach().cpu().numpy().tolist()
        return dict(zip(texts, scores))

    @staticmethod
    def _postprocess(features: torch.Tensor, normalize: bool) -> np.ndarray:
        if normalize:
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).detach().cpu().numpy()


def load_clip_generate_loader(config) -> EmbeddingModelLoader:
    """Same pattern as `load_bot_sort_tracker`: builds a CLIP loader from config."""
    cfg = _build_clip_generate_cfg(config) if isinstance(config, dict) else config
    return EmbeddingModelLoader(
        model_family=cfg.model_family,
        model_version=cfg.model_version,
        normalize=cfg.normalize,
        device=cfg.device,
    )


def load_clip_comparison_loader(config) -> EmbeddingModelLoader:
    cfg = _build_clip_comparison_cfg(config) if isinstance(config, dict) else config
    return EmbeddingModelLoader(
        model_family=cfg.model_family,
        model_version=cfg.model_version,
        normalize=cfg.normalize,
        device=cfg.device,
    )


def load_perception_encoder_loader(config) -> EmbeddingModelLoader:
    cfg = _build_perception_encoder_cfg(config) if isinstance(config, dict) else config
    return EmbeddingModelLoader(
        model_family=cfg.model_family,
        model_version=cfg.model_version,
        normalize=cfg.normalize,
        device=cfg.device,
    )
