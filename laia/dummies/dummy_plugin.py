from pytorch_lightning.plugins.ddp_plugin import DDPPlugin

import laia.common.logging as log


class DummyLoggingPlugin(DDPPlugin):
    def __init__(self, log_filepath):
        super().__init__()
        self.log_filepath = log_filepath
        self.setup_logging(self.log_filepath)

    @staticmethod
    def setup_logging(log_filepath):
        log.config(fmt="%(message)s", filepath=log_filepath, overwrite=True)

    def configure_ddp(self, *args, **kwargs):
        # call _setup_logging again here otherwise processes
        # spawned by multiprocessing are not correctly configured
        self.setup_logging(self.log_filepath)
        return super().configure_ddp(*args, **kwargs)

    def __del__(self):
        log.clear()
