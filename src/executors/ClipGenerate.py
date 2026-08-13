"""
DEBUG TEST 4 -- torch/open_clip_torch tamamen cikarilmis, saf minimal
setup.py. Amac: build/deploy asamasinin agir bagimliliklarda (torch ~2GB)
timeout/basarisiz olma ihtimalini izole etmek.

KULLANIM: Bu icerigi setup.py'nin uzerine koy, requirements_MINIMAL.txt'yi
de requirements.txt'nin uzerine koy. ClipGenerate.py'yi Test 3'teki
(DEBUG_MARKER, orijinal Capsule versiyonu) haline dondur -- yani SADECE
bagimliliklari degistiriyoruz, kod ayni kalsin ki neyin etkili oldugunu
ayirt edebilelim.
"""

import setuptools

setuptools.setup(
    name="novavision-cap-embedding-extraction",
    version="0.0.1",
    author="DigiNova",
    author_email="info@diginova.com.tr",
    description="NOVAVISION Embedding Extraction (DEBUG: minimal deps)",
    license="MIT",
    install_requires=[
        "pydantic>=1.10,<2.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=[
        "novavision.cap.embedding-extraction",
        "novavision.cap.embedding-extraction.executors",
        "novavision.cap.embedding-extraction.models",
        "novavision.cap.embedding-extraction.utils",
    ],
    package_dir={"novavision.cap.embedding-extraction": "src"},
    python_requires=">=3.9",
)