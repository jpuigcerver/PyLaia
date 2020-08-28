import os

import setuptools


def get_scripts():
    return [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), script)
        for script in (
            "pylaia-htr-create-model",
            "pylaia-htr-decode-ctc",
            "pylaia-htr-train-ctc",
            "pylaia-htr-netout",
        )
    ]


def get_requirements():
    requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    with open(requirements_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


setuptools.setup(
    name="laia",
    use_scm_version={
        "write_to": "laia/version.py",
        "write_to_template": '__version__ = "{version}"\n',
        "local_scheme": lambda v: "+{}.{}{}".format(
            v.node, v.branch, ".dirty" if v.dirty else ""
        ),
    },
    author="Joan Puigcerver",
    author_email="joapuipe@gmail.com",
    license="MIT",
    url="https://github.com/jpuigcerver/PyLaia",
    # Requirements
    setup_requires=["setuptools_scm"],
    install_requires=get_requirements(),
    extras_require={
        "dev": ["pre-commit", "isort", "black", "setuptools_scm"],
        "test": ["pytest", "parameterized"],
    },
    python_requires=">=3.6",
    # Package contents
    packages=setuptools.find_packages(),
    scripts=get_scripts(),
)
