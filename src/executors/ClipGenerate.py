"""
DEBUG TEST 3 -- Capsule yerine Component base sinifini deniyoruz.

Ugur Hoca'nin platformda birakti gi sablon dosyada (novavision-ai/cap-embedding,
Initial commit) su desen kullaniliyordu:

    from sdks.novavision.src.base.component import Component   <- Capsule DEGIL
    class Package(Component):                                   <- Capsule DEGIL

Bizim kodumuz (ve referans ObjectTracking) her yerde Capsule kullaniyor.
Eger platformdaki guncel SDK'da Capsule sinifi artik yoksa (Component'e
donusmusse), "from sdks.novavision.src.base.capsule import Capsule" satiri
dosyanin en tepesinde, run()'a hic ulasmadan ModuleNotFoundError ile patlar --
bu da onceki 3 testimizin (gercek CLIP, sahte embedding, "sadece raise")
neden UCUNUN DE ayni sekilde sessiz kaldigini acikliyor olabilir.

KULLANIM: src/executors/ClipGenerate.py'nin icerigini GECICI olarak
bununla degistir, commit+push et, platformda senkronize et, Run Flow,
LOG ekranina bak.

Bu SADECE Capsule/Component farkini izole ediyor -- requirements.txt/
setup.py'ye HIC DOKUNMA, sadece bu .py dosyasini degistir. Ikinci testi
(minimal requirements.txt) bununla AYNI ANDA YAPMA, sonucu ayirt edemeyiz.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor


class ClipGenerate(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def run(self):
        raise Exception("DEBUG_MARKER_COMPONENT: run() reached via Component base class")


if "__main__" == __name__:
    Executor(sys.argv[1]).run()