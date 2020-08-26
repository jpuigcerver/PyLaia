from typing import Any


class EngineException(Exception):
    def __init__(
        self, epoch: int, global_step: int, batch: Any, cause: Exception = None
    ):
        self._epoch = epoch
        self._global_step = global_step
        self._batch = batch
        self._cause = cause

    def __str__(self):
        exception_text = f'"{repr(self._cause)}" ' if self._cause else ""
        return (
            f"Exception {exception_text}raised during epoch={self._epoch}, "
            f"global_step={self._global_step} with batch={self._batch}"
        )
