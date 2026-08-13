"""
GEÇİCİ TEST VERSİYONU -- torch/open_clip'i hiç import etmez.
Amaç: bootstrap()/run() akışının kendisinin (sdks.novavision entegrasyonu,
Response inşası) çalışıp çalışmadığını, torch/open_clip'ten BAĞIMSIZ olarak
görmek. Eğer bu versiyon çalışıp bir çıktı üretirse, sorunun kesinlikle
torch/open_clip_torch kurulumunda olduğu kanıtlanmış olur.

KULLANIM: Bu dosyanın içeriğini src/executors/ClipGenerate.py'nin
ÜZERİNE geçici olarak kopyala, commit+push et, platformda paketi
yeniden yükle/senkronize et, Run Flow'u dene. Sonucu paylaş, sonra
orijinal ClipGenerate.py'ye geri dön (git checkout ile).
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.base.capsule import Capsule
from sdks.novavision.src.helper.executor import Executor
# DİKKAT: utils.py import edilmiyor -- torch/open_clip hiç yüklenmiyor.

from capsules.EmbeddingExtraction.src.models.PackageModel import PackageModel
from capsules.EmbeddingExtraction.src.utils.response import build_clip_generate_response


class ClipGenerate(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.input_data = self.request.get_param("inputData")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        # Gerçek model yüklemiyor -- sadece bootstrap()'ın çağrılıp
        # çağrılmadığını, hata verip vermediğini test ediyoruz.
        return {"loader": None}

    def run(self):
        # Sahte, sabit bir embedding dönüyoruz -- torch/open_clip'e hiç dokunmadan.
        fake_embedding = [0.1] * 512

        meta = {
            "model_family": "DEBUG",
            "model_version": "fake",
            "input_type": "unknown",
            "embedding_dim": 512,
        }

        packageModel = build_clip_generate_response(
            context=self,
            embedding=fake_embedding,
            meta=meta,
        )
        return packageModel


if "__main__" == __name__:
    Executor(sys.argv[1]).run()