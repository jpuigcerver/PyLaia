import inspect
import os
from collections import deque
from typing import Any, Callable, Optional

import torch

from laia.common.logging import get_logger
from laia.common.random import get_rng_state

_logger = get_logger(__name__)


class Saver:
    def __call__(self, *args: Any, **kwargs: Any):
        return self.save(*args, **kwargs)

    def save(self, *args: Any, **kwargs: Any):
        raise NotImplementedError


class BasicSaver(Saver):
    def save(self, obj: Any, filepath: str) -> str:
        dirname = os.path.dirname(os.path.normpath(filepath))
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)
        torch.save(obj, filepath)
        return filepath


class ObjectSaver(Saver):
    def __init__(self, filepath: str) -> None:
        self._filepath = filepath
        self._basic_saver = BasicSaver()

    def save(self, func_or_class: Callable, *args: Any, **kwargs: Any) -> str:
        return self._basic_saver.save(
            {
                "module": inspect.getmodule(func_or_class).__name__,
                "name": func_or_class.__name__,
                "args": args,
                "kwargs": kwargs,
            },
            self._filepath,
        )


class ModelSaver(ObjectSaver):
    def __init__(self, save_path: str, filename: str = "model") -> None:
        super().__init__(os.path.join(save_path, filename))

    def save(self, func: Callable, *args: Any, **kwargs: Any) -> str:
        path = super().save(func, *args, **kwargs)
        _logger.debug("Saved model {}", path)
        return path


class CheckpointSaver(Saver):
    def __init__(self, filepath: str) -> None:
        self._filepath = filepath
        self._basic_saver = BasicSaver()

    def get_ckpt(self, suffix: str) -> str:
        return (
            "{}-{}".format(self._filepath, suffix)
            if suffix is not None
            else self._filepath
        )

    def save(self, state: Any, suffix: Optional[str] = None) -> str:
        path = self._basic_saver.save(state, self.get_ckpt(suffix))
        _logger.debug("Saved checkpoint {}", path)
        return path


class ModelCheckpointSaver(Saver):
    def __init__(self, ckpt_saver: CheckpointSaver, model: torch.nn.Module) -> None:
        self._ckpt_saver = ckpt_saver
        self._model = model

    def save(self, suffix: Optional[str] = None) -> str:
        return self._ckpt_saver.save(self._model.state_dict(), suffix=suffix)


class StateCheckpointSaver(Saver):
    def __init__(
        self,
        ckpt_saver: CheckpointSaver,
        obj: Any,
        device: Optional[torch.device] = None,
    ) -> None:
        self._ckpt_saver = ckpt_saver
        self._obj = obj
        self._device = device

    def save(self, suffix: Optional[str] = None) -> str:
        state = self._obj.state_dict()
        state["rng"] = get_rng_state(device=self._device)
        return self._ckpt_saver.save(state, suffix=suffix)


class RollingSaver(Saver):
    """Saver wrapper that keeps a maximum number of files"""

    def __init__(self, saver: Saver, keep: int = 5) -> None:
        assert keep > 0
        self._saver = saver
        self._keep = keep
        self._last_saved = deque()  # type: deque

    def save(self, *args: Any, **kwargs: Any) -> str:
        path = self._saver.save(*args, **kwargs)
        if len(self._last_saved) >= self._keep:
            last = self._last_saved.popleft()
            try:
                os.remove(last)
                _logger.debug("{} checkpoint removed", last)
            except OSError:
                # Someone else removed the checkpoint, not a big deal
                pass
        self._last_saved.append(path)
        return path
