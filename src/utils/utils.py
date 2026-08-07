"""
utils.py

CLIP / Perception Encoder model yükleme ve embedding çıkarım yardımcıları.
`BoTSORTTracker/reid.py`'deki appearance-extractor deseniyle (cropping,
batching, normalisation, caching) aynı ruhta yazılmıştır.

Notlar:
- CLIP tarafı `open_clip_torch` paketiyle tam çalışır durumdadır
  (pip install open_clip_torch); GPU şart değildir, CPU'da da çalışır.
- Perception Encoder tarafı Meta'nın `perception_models` paketine
  bağımlıdır ve GPU/CUDA zorunludur (bkz. Roboflow'un
  roboflow_core/perception_encoder@v1 blok dokümantasyonu). Bu ortamda
  paket kurulu olmadığından en iyi çaba (best-effort) yazılmıştır.
"""

from types import SimpleNamespace

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


def _build_clip_cfg(config):
    """`_build_bot_sort_cfg` ile aynı desende: Application().get_param() ile
    ConfigClipAdvance toggle'ını okur; True ise alt config'leri, False ise
    varsayılanları kullanır. Config değerleri SADECE burada, bootstrap()
    içinde okunur -- request.get_param() yalnızca inputs için kullanılır."""
    application = Application()
    advance = application.get_param(config=config, name="ConfigClipAdvance")

    if advance == "True":
        version = application.get_param(config=config, name="ClipVersion")
        normalize = application.get_param(config=config, name="ClipNormalize")
        return SimpleNamespace(
            model_family="CLIP",
            model_version=version or "ViT-B-16",
            normalize=normalize if normalize is not None else True,
            device="cpu",  # CLIP için GPU şart değil; CUDA varsa otomatik kullanılır
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
            device="cuda",  # ⚠ Perception Encoder GPU/CUDA zorunlu
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
                f"Bilinmeyen CLIP versiyonu: '{model_version}'. "
                f"Desteklenenler: {list(CLIP_VERSION_TO_OPEN_CLIP.keys())}"
            )
    elif model_family == "PerceptionEncoder":
        if model_version not in PERCEPTION_ENCODER_VERSIONS:
            raise ValueError(
                f"Bilinmeyen Perception Encoder versiyonu: '{model_version}'. "
                f"Desteklenenler: {sorted(PERCEPTION_ENCODER_VERSIONS)}"
            )
    else:
        raise ValueError(f"Bilinmeyen model ailesi: '{model_family}'")
    return model_version


class EmbeddingModelLoader:
    """CLIP veya Perception Encoder modelini yükleyip cache'te tutan sınıf.
    `bootstrap()` içinde bir kez oluşturulur, `run()` çağrılarında tekrar
    kullanılır (reid.py'deki ReID sınıfıyla aynı yaşam döngüsü).

    `normalize`, config'ten bootstrap-time'da okunup burada sabitlenir --
    çalışma zamanında (`run()` içinde) tekrar config'ten okunmaz, çünkü
    referans SDK deseninde config'ler yalnızca bootstrap() içinde
    Application().get_param() ile okunur (bkz. utils.py:_build_clip_cfg)."""

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
            raise ValueError(f"Bilinmeyen model ailesi: '{self.model_family}'")

    def _load_clip(self):
        import open_clip  # pip install open_clip_torch

        model_name, pretrained = CLIP_VERSION_TO_OPEN_CLIP[self.model_version]
        # OpenAI'nin orijinal CLIP ağırlıkları (ViT-B-16/32, RN50) QuickGELU
        # aktivasyonuyla eğitilmiştir (open_clip.get_pretrained_cfg(...)["quick_gelu"]
        # == True). force_quick_gelu verilmezse model varsayılan (standart) GELU ile
        # kurulur ve ağırlıklar yanlış aktivasyon fonksiyonuyla yüklenmiş olur --
        # forward pass çalışır ama embedding'ler eğitim zamanındakinden sapar.
        # Doğrulama: force_quick_gelu=False -> nn.GELU, True -> nn.QuickGELU
        # (bu ortamda open_clip.get_pretrained_cfg ile ve katman tipini
        # inceleyerek test edildi).
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, force_quick_gelu=True
        )
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def _load_perception_encoder(self):
        """⚠ Bu ortamda doğrulanamadı — bkz. modül docstring'i."""
        try:
            import core.vision_encoder.pe as pe  # type: ignore
            import core.vision_encoder.transforms as pe_transforms  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Perception Encoder paketi bulunamadı. Kurulum: "
                "pip install git+https://github.com/facebookresearch/perception_models.git "
                "-- GPU/CUDA zorunludur."
            ) from exc

        model = pe.CLIP.from_config(self.model_version, pretrained=True)
        self.model = model.to(self.device).eval()
        self.preprocess = pe_transforms.get_image_transform(model.image_size)
        self.tokenizer = pe_transforms.get_text_tokenizer(model.context_length)

    @torch.no_grad()
    def embed_image(self, image_array: np.ndarray, normalize: bool = None) -> np.ndarray:
        """NumPy (H, W, 3) RGB array -> embedding vektörü.
        `normalize` verilmezse (None), bootstrap-time'da config'ten
        okunan `self.normalize` kullanılır."""
        pil_image = PILImage.fromarray(image_array.astype("uint8"), "RGB")
        tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        features = self.model.encode_image(tensor)
        return self._postprocess(features, self.normalize if normalize is None else normalize)

    @torch.no_grad()
    def embed_text(self, text: str, normalize: bool = None) -> np.ndarray:
        """Serbest metin -> embedding vektörü."""
        tokens = self.tokenizer([text]).to(self.device)
        features = self.model.encode_text(tokens)
        return self._postprocess(features, self.normalize if normalize is None else normalize)

    @staticmethod
    def _postprocess(features: torch.Tensor, normalize: bool) -> np.ndarray:
        if normalize:
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).detach().cpu().numpy()


def load_clip_loader(config) -> EmbeddingModelLoader:
    """`load_bot_sort_tracker` ile aynı desende: config'ten CLIP loader'ı kurar."""
    cfg = _build_clip_cfg(config) if isinstance(config, dict) else config
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
