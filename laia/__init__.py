import subprocess
import sys
from pathlib import Path
from shutil import which
from typing import List

import laia.callbacks
import laia.data
import laia.decoders
import laia.engine
import laia.loggers
import laia.losses
import laia.nn
import laia.utils

__all__ = ["__version__", "__root__", "get_installed_versions"]
__lib__ = Path(__file__).parent
__root__ = lib.parent
__version__ = (__lib__ / "VERSION").read_text()

try:
    branch = subprocess.check_output(
        [which("git"), "-C", str(__root__), "branch", "--show-current"],
        stderr=subprocess.DEVNULL,
    )
    branch = branch.decode().strip()
    node = subprocess.check_output(
        [which("git"), "-C", str(__root__), "describe", "--always", "--dirty"],
        stderr=subprocess.DEVNULL,
    )
    node = node.decode().strip()
    __version__ += f"-{branch}-{node}"
except subprocess.CalledProcessError:
    pass


def get_installed_versions() -> List[str]:
    requirements_path = __root__ / "requirements.txt"
    if not requirements_path.exists():
        return []
    requirements = []
    with open(requirements_path) as f:
        for r in f.readlines():
            r = r.strip()
            r = r.split(" @ ")[0]  # support 'pkg @ git+https://...' notation
            r = r.split("==")[0]
            r = r.split(">=")[0]
            r = r.split("<")[0]
            r = r.split("[")[0]  # support 'pkg[extras]' notation
            requirements.append(r)
    freeze = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze", "--exclude-editable"]
    )
    freeze = freeze.decode().strip().split("\n")
    versions = [
        r
        for r in freeze
        if r in requirements
        or ("==" in r and r[: r.index("==")] in requirements)
        or (" @ " in r and r[: r.index(" @ ")] in requirements)
    ]
    versions.append(f"laia=={__version__}")
    return versions
