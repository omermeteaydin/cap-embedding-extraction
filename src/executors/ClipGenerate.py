"""
DEBUG SEVIYE 2 -- yalnizca sdks.novavision.base.capsule.Capsule import
ediliyor, gercek response mekanizmasi (PackageHelper) BILE kullanilmiyor.
Amac: import zincirinde NEREDE patladigini daraltmak.
 
run() bilerek bir Exception firlatiyor -- eger bu LOG ekraninda
gorunurse, en azindan import + bootstrap + run() cagrisina kadar
sorunsuz geldigimizi kanitlar (sorun response insasinda demektir).
Eger LOG'da hicbir sey gorunmezse (hala "Env Exited" ya da bomboş),
sorun import/init asamasinda demektir.
 
KULLANIM: src/executors/ClipGenerate.py'nin icerigini GECICI olarak
bununla degistir, commit+push et, PLATFORMDA PAKETI YENIDEN
SENKRONIZE ET (onemli!), Run Flow'u dene, LOG ekranina bak.
"""
 
import os
import sys
 
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))
 
from sdks.novavision.src.base.capsule import Capsule
from sdks.novavision.src.helper.executor import Executor
 
 
class ClipGenerate(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        # PackageModel(**request.data) parse etmiyoruz bile -- bunu da
        # izolasyon disi biraktik.
 
    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}
 
    def run(self):
        # Bilerek patlatiyoruz -- LOG ekraninda bu mesaji gorup
        # gormedigimiz, buraya kadar gelip gelmedigimizi soyler.
        raise Exception("DEBUG_MARKER: run() reached successfully")
 
 
if "__main__" == __name__:
    Executor(sys.argv[1]).run()