import os
from collections import OrderedDict
from glob import glob
from importlib import import_module
from io import BytesIO
from typing import Any, Callable, Optional, Union

import natsort as ns
import pytorch_lightning as pl
import torch

from laia.common.logging import get_logger

_logger = get_logger(__name__)


class Loader:
    def __call__(self, *args: Any, **kwargs: Any):
        return self.load(*args, **kwargs)

    def load(self, *args: Any, **kwargs: Any):
        raise NotImplementedError


class BasicLoader(Loader):
    def load(
        self, f: Union[str, BytesIO], device: Optional[Union[str, torch.device]] = None
    ) -> Any:
        try:
            return torch.load(f, map_location=device)
        except FileNotFoundError:
            _logger.info("Could not find the file {}", f)


class ObjectLoader(Loader):
    def __init__(
        self, f: Union[str, BytesIO], device: Optional[Union[str, torch.device]] = None
    ) -> None:
        self._f = f
        self._device = device
        self._loader = BasicLoader()

    def load(self) -> Any:
        obj = self._loader.load(self._f, device=self._device)
        if obj is None:
            return
        module = import_module(obj["module"])
        fn = getattr(module, obj["name"])
        args = obj.get("args", [])
        kwargs = obj.get("kwargs", {})
        return fn(*args, **kwargs)


class ModelLoader(ObjectLoader):
    def __init__(
        self,
        load_path: str,
        filename: str = "model",
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self._path = os.path.join(load_path, filename)
        super().__init__(self._path, device=device)

    def load(self) -> Any:
        model = super().load()
        if model is not None:
            _logger.info("Loaded model {}", self._path)
        return model

    def get_model_state_dict(self, checkpoint: str) -> OrderedDict:
        ckpt = self._loader.load(checkpoint, device=self._device)
        if "pytorch-lightning_version" in ckpt:
            _logger.debug(
                "Checkpoint trained for {} epochs, {} iterations",
                ckpt["epoch"],
                ckpt["global_step"],
            )
            state_dict = ckpt["state_dict"]
            assert all(k.startswith("model.") for k in state_dict.keys())
            return OrderedDict((k[len("model.") :], v) for k, v in state_dict.items())
        if "tr_engine" in ckpt:
            # backwards compatibility
            engine = ckpt["tr_engine"]
            _logger.debug(
                "Checkpoint trained for {} epochs, {} iterations",
                engine["epochs"],
                engine["iterations"],
            )
            return engine["model"]
        return ckpt

    @staticmethod
    def choose_by(
        pattern: str, key: Optional[Callable] = None, reverse: bool = True
    ) -> Optional[str]:
        matches = glob(pattern)
        matches = [m for m in matches if os.path.isfile(m)]
        if not matches:
            return
        return ns.natsorted(matches, key=key, reverse=reverse, alg=ns.ns.PATH)[0]

    def load_by(self, checkpoint: str) -> Optional[torch.nn.Module]:
        _logger.info('Using checkpoint "{}"', checkpoint)
        model = self.load()
        if model is not None:
            state_dict = self.get_model_state_dict(checkpoint)
            model.load_state_dict(state_dict)
        return model

    @staticmethod
    def find_best(directory: str, monitor: str, mode: str = "min") -> Optional[str]:
        ckpts = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.endswith(".ckpt")
        ]
        ckpts = [(f, torch.load(f, map_location="cpu")) for f in ckpts]
        ckpts = [
            (f, ckpt["callbacks"][type(pl.callbacks.ModelCheckpoint())])
            for f, ckpt in ckpts
        ]
        # note: requires checkpoints generated using pl>1.0.4
        ckpts = [(f, ckpt) for f, ckpt in ckpts if ckpt.get("monitor") == monitor]
        if not ckpts:
            return
        mode = min if mode == "min" else max
        f, _ = mode(ckpts, key=lambda x: x[1]["current_score"])
        return f

    @staticmethod
    def prepare_checkpoint(checkpoint: str, exp_dirpath: str, monitor: str) -> str:
        if checkpoint:
            checkpoint_path = os.path.join(exp_dirpath, checkpoint)
            found = ModelLoader.choose_by(checkpoint_path)
            err_msg = f'Could not find the checkpoint "{checkpoint_path}"'
        else:
            found = ModelLoader.find_best(exp_dirpath, monitor)
            err_msg = f'Could not find a valid checkpoint in "{exp_dirpath}"'
        assert found, err_msg
        return found
