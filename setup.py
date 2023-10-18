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
    url="https://github.com/jpuigcerver/PyLaia",
    download_url="https://github.com/jpuigcerver/PyLaia",
    # Requirements
    setup_requires=["setuptools_scm"],
    install_requires=get_requirements("requirements.txt"),
    extras_require={
        "dev": ["pre-commit", "isort", "black"],
        "test": ["pytest", "pytest-cov", "pandas", "regex"],
        "docs": get_requirements("doc-requirements.txt"),
    },
    python_requires=">=3.6",
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
)
