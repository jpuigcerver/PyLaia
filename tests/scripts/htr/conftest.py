import os
import ssl
import subprocess
import sys
from typing import List, Optional, Tuple

import pytest
from torchvision.datasets.utils import download_url

from laia import __root__


def call_script(
    file: str, args: List[str], timeout: Optional[int] = 60 * 3
) -> Tuple[str, str]:
    # To test distributed without GPUs, we are using the ddp_cpu accelerator."
    # ddp_cpu uses multiprocessing.spawn behind the scenes. Spawned children do not"
    # inherit all parent resources, consequently, mocking does not work because"
    # children do not have their mocks configured. This means that we can't call"
    # the script's main manually with std{out,err} redirected (e.g. with"
    # contextlib.redirect_stdout) as if we were using only one process. There is"
    # also the issue that spawned processes' objects must be entirely pickleable."
    # To circumvent these limitations, we launch our own process (here) to call the"
    # script file and capture its stdout and stderr"
    args = [str(a) for a in args]
    command = [sys.executable, file] + args
    print(" ".join(command))
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        stdout, stderr = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        stdout, stderr = p.communicate()
    stdout = stdout.decode("utf-8")
    stderr = stderr.decode("utf-8")
    return stdout, stderr


@pytest.fixture
def downloader():
    """https://docs.pytest.org/en/latest/fixture.html#conftest-py-sharing-fixture-functions"""

    def get(resource: str, tmpdir: str) -> str:
        # --no-check-certificate
        ssl._create_default_https_context = ssl._create_unverified_context

        from_root = "https://huggingface.co/Teklia/pylaia-huginmunin/resolve/main"
        to_root = tmpdir / resource

        url = from_root + "/" + resource
        if not os.path.exists(to_root):
            download_url(url, str(tmpdir))
        return str(to_root)

    return get
