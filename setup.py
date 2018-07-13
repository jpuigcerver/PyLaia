from setuptools import setup, find_packages

setup(
    name="laia",
    version="0.1",
    author="Joan Puigcerver",
    author_email="joapuipe@gmail.com",
    license="MIT",
    url="https://github.com/jpuigcerver/PyLaia",
    # Requirements
    install_requires=[
        "cffi>=1.0",
        "editdistance",
        "future",
        'mock;python_version<"3.0"',
        "numpy",
        "scipy",
        "tqdm",
        "torch==0.4",
        'typing;python_version<"3.5"',
        "Pillow",
    ],
    # Package contents
    packages=find_packages(),
    scripts=[
        "pylaia-htr-create-model",
        "pylaia-htr-decode-ctc",
        "pylaia-htr-train-ctc",
        "pylaia-htr-netout",
    ],
)
