from pathlib import Path
from typing import List

import setuptools

DIR = Path(__file__).parent

VERSION = (DIR / "laia" / "VERSION").read_text()


def get_requirements(filename: str) -> List[str]:
    return [
        line.strip()
        for line in (DIR / filename).read_text(encoding="utf-8").splitlines()
    ]


def get_long_description() -> str:
    return (DIR / "README.md").read_text(encoding="utf-8")


setuptools.setup(
    name="pylaia",
    version=VERSION,
    author="Joan Puigcerver",
    author_email="joapuipe@gmail.com",
    maintainer="Teklia",
    maintainer_email="contact@teklia.com",
    license="MIT",
    license_file="LICENSE",
    url="https://atr.pages.teklia.com/pylaia/",
    download_url="https://gitlab.teklia.com/atr/pylaia",
    project_urls={
        "Documentation": "https://atr.pages.teklia.com/pylaia/",
        "Source": "https://gitlab.teklia.com/atr/pylaia/",
        "Tracker": "https://gitlab.teklia.com/atr/pylaia/issues/",
    },
    # Requirements
    setup_requires=["setuptools_scm"],
    install_requires=get_requirements("requirements.txt"),
    extras_require={
        "dev": ["pre-commit", "isort", "black"],
        "test": ["tox"],
        "docs": get_requirements("doc-requirements.txt"),
    },
    python_requires=">= 3.9, < 3.11",
    # Package contents
    packages=setuptools.find_packages(exclude=["tests"]),
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "pylaia-htr-create-model=laia.scripts.htr.create_model:main",
            "pylaia-htr-train-ctc=laia.scripts.htr.train_ctc:main",
            "pylaia-htr-decode-ctc=laia.scripts.htr.decode_ctc:main",
            "pylaia-htr-netout=laia.scripts.htr.netout:main",
        ],
    },
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: MIT License",
        # Specify the Python versions you support here.
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        # Topics
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Text Processing :: Linguistic",
    ],
    keywords="HTR OCR python",
)
