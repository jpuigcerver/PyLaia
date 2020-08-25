__all__ = [
    "common",
    "conditions",
    "data",
    "decoders",
    "engine",
    "experiments",
    "hooks",
    "losses",
    "losses",
    "meters",
    "models",
    "nn",
    "utils",
    "__version__",
    "get_installed_versions",
]

from typing import List

import laia.common
import laia.conditions
import laia.data
import laia.decoders
import laia.engine
import laia.experiments
import laia.hooks
import laia.losses
import laia.meters
import laia.models
import laia.nn

try:
    from laia.version import __version__
except ImportError:
    from setuptools_scm import get_version

    __version__ = get_version(
        root="..",
        relative_to=__file__,
        local_scheme=lambda v: "+{}.{}{}".format(
            v.node, v.branch, ".dirty" if v.dirty else ""
        ),
    )


def get_installed_versions() -> List[str]:
    import subprocess

    with open("requirements.txt") as f:
        requirements = [r.strip() for r in f.readlines()]
    freeze = subprocess.check_output(["pip", "freeze", "--exclude-editable"])
    freeze = freeze.decode("ascii").strip().split()
    versions = [
        r for r in freeze if r in requirements or r[: r.index("==")] in requirements
    ]
    versions.append(f"laia=={__version__}")
    return versions
