from typing import List

import laia.callbacks
import laia.data
import laia.decoders
import laia.engine
import laia.loggers
import laia.losses
import laia.nn
import laia.utils

__all__ = ["__version__", "get_installed_versions"]

try:
    from setuptools_scm import get_version

    __version__ = get_version(
        root="..",
        relative_to=__file__,
        local_scheme=lambda v: f"+{v.node}.{v.branch}{'.dirty' if v.dirty else ''}",
    )
except ImportError:
    from laia.version import __version__


def get_installed_versions() -> List[str]:
    import subprocess
    from pathlib import Path

    requirements_path = Path(__file__).parent.parent / "requirements.txt"
    with open(requirements_path) as f:
        requirements = [r.strip() for r in f.readlines()]
    freeze = subprocess.check_output(["pip", "freeze", "--exclude-editable"])
    freeze = freeze.decode("ascii").strip().split()
    versions = [
        r
        for r in freeze
        if r in requirements or ("==" in r and r[: r.index("==")] in requirements)
    ]
    versions.append(f"laia=={__version__}")
    return versions
