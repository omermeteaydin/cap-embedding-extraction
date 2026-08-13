"""
TEST 2 -- ClipGenerate.py (Capsule versiyonu, minimal bagimliliklarla test icin)

Bu dosya src/executors/ClipGenerate.py'nin UZERINE konulacak.
Capsule kullaniyor (Component degil -- o teori Test 1'de elendi).
run() bilerek patliyor, LOG'da bu mesaji arayacagiz.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.base.capsule import Capsule
from sdks.novavision.src.helper.executor import Executor


class ClipGenerate(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def run(self):
        raise Exception("DEBUG_MARKER_MINIMAL_DEPS: run() reached with minimal requirements")


if "__main__" == __name__:
    Executor(sys.argv[1]).run()