import os

import setuptools

DIR = os.path.dirname(os.path.abspath(__file__))


def get_requirements():
    requirements_path = os.path.join(DIR, "requirements.txt")
    with open(requirements_path, encoding="utf-8") as f:
        return [line.strip() for line in f]


def get_long_description():
    readme_path = os.path.join(DIR, "README.md")
    return open(readme_path, encoding="utf-8").read()


setuptools.setup(
    name="laia",
    use_scm_version={
        "write_to": "laia/version.py",
        "write_to_template": '__version__ = "{version}"\n',
        "local_scheme": lambda v: f"+{v.node}.{v.branch}{'.dirty' if v.dirty else ''}",
    },
    author="Joan Puigcerver",
    author_email="joapuipe@gmail.com",
    maintainer="Carlos Mocholí",
    maintainer_email="carlossmocholi@gmail.com",
    license="MIT",
    url="https://github.com/jpuigcerver/PyLaia",
    download_url="https://github.com/jpuigcerver/PyLaia",
    # Requirements
    setup_requires=["setuptools_scm"],
    install_requires=get_requirements(),
    extras_require={
        "dev": ["pre-commit", "isort", "black", "setuptools_scm"],
        "test": ["pytest", "pytest-cov", "pandas"],
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
