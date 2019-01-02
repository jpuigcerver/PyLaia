from __future__ import absolute_import

import io
import os
import subprocess

import setuptools
import setuptools.command.build_py

cwd = os.path.dirname(os.path.abspath(__file__))


def _git_output(args):
    stderr = io.open(os.devnull, "w", encoding="utf-8")
    output = None
    try:
        output = subprocess.check_output(args, cwd=cwd).decode("ascii").strip()
    finally:
        stderr.close()
        return output


def git_commit(short=False):
    """Returns the hash of the current Git commit, or None
    if the package is not under Git."""
    args = ["git", "rev-parse", "HEAD"]
    if short:
        args = args[0:2] + ["--short"] + args[2:]
    return _git_output(args)


def git_branch():
    """Returns the name of the current Git branch, or None
    if the package is not under Git."""
    return _git_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])


def git_is_dirty():
    """Returns whether the repository contains local changes or not"""
    return bool(_git_output(["git", "status", "--short"]))


MAJOR = 0
MINOR = 1
MICRO = 0
VERSION = "{}.{}.{}".format(MAJOR, MINOR, MICRO)


class create_version_file(setuptools.Command):
    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        print("creating version file")
        version_path = os.path.join(cwd, "laia", "version.py")
        with io.open(version_path, "w", encoding="utf-8") as f:
            full_version = "{}+{}{}".format(
                VERSION, git_commit(short=True), "-dirty" if git_is_dirty() else ""
            )
            f.write("__full_version__ = '{}'\n".format(full_version))
            f.write("__version__ = '{}'\n".format(VERSION))
            f.write("__commit__ = '{}'\n".format(git_commit()))
            f.write("__branch__ = '{}'\n".format(git_branch()))


class build_py(setuptools.command.build_py.build_py):
    def run(self):
        self.run_command("create_version_file")
        setuptools.command.build_py.build_py.run(self)


def get_scripts():
    return [
        os.path.join(cwd, script)
        for script in (
            "pylaia-htr-create-model",
            "pylaia-htr-decode-ctc",
            "pylaia-htr-train-ctc",
            "pylaia-htr-netout",
        )
    ]


def get_requirements():
    requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    with io.open(requirements_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


setuptools.setup(
    name="laia",
    version=VERSION,
    author="Joan Puigcerver",
    author_email="joapuipe@gmail.com",
    license="MIT",
    url="https://github.com/jpuigcerver/PyLaia",
    # Requirements
    install_requires=get_requirements(),
    # Package contents
    packages=setuptools.find_packages(),
    scripts=get_scripts(),
    cmdclass={"create_version_file": create_version_file, "build_py": build_py},
)
