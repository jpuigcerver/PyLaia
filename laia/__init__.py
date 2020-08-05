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
]

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
