from os.path import join
from typing import Any, Dict

import laia.common.logging as log
from laia import get_installed_versions


def common_main(args: Dict[str, Any]) -> Dict[str, Any]:
    del args["config"]
    # configure logging
    logging = args.pop("logging")
    if logging["filepath"] is not None:
        logging["filepath"] = join(
            args["common"].experiment_dirpath, logging["filepath"]
        )
    log.config(**logging)
    log.info(f"Arguments: {args}")
    versions = get_installed_versions()
    if versions:
        log.info("Installed:", versions)
    return args
