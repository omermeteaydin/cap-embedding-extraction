import setuptools

setuptools.setup(
    name="novavision-cap-embedding-extraction",
    version="0.0.1",
    author="DigiNova",
    author_email="info@diginova.com.tr",
    description="NOVAVISION Embedding Extraction (CLIP / Perception Encoder)",
    license="MIT",
    install_requires=[
        "torch>=2.0.0",
        "open_clip_torch>=2.24.0",
        "numpy",
        "Pillow",
        # Perception Encoder is optional (requires GPU, separate install):
        # git+https://github.com/facebookresearch/perception_models.git
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