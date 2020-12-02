import inspect
import os
from typing import Any, Callable

import torch

from laia.common.logging import get_logger

_logger = get_logger(__name__)


class Saver:
    def __call__(self, *args: Any, **kwargs: Any):
        return self.save(*args, **kwargs)

    def save(self, *args: Any, **kwargs: Any):
        raise NotImplementedError


class BasicSaver(Saver):
    def save(self, obj: Any, filepath: str) -> str:
        filepath = os.path.realpath(filepath)
        dirname = os.path.dirname(filepath)
        os.makedirs(dirname, exist_ok=True)
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
