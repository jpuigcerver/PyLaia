from contextlib import contextmanager
from typing import Iterable, Callable, Union, Optional, Any

import torch
from tqdm import tqdm

import laia
import laia.common.logging as log
from laia.engine.engine_exception import EngineException
from laia.hooks import action

_logger = log.get_logger(__name__)

EPOCH_START = "EPOCH_START"
EPOCH_END = "EPOCH_END"
ITER_START = "ITER_START"
ITER_END = "ITER_END"


class Engine:
    """Wrapper class to train a model.

    Args:
        model: model to train.
        data_loader: iterable object from which batches are read.
            (default: None)
        batch_input_fn (optional): function used to extract the input
            for the model (e.g. a ``torch.Tensor``), from the batch loaded by
            the ``data_loader``. If ``None``, the batch is fed as-is to the
            model. (default: None)
        batch_target_fn (optional): if given, this callable object
            is used to extract the targets from the batch, which are
            passed to the `ITER_START` and `ITER_END` hooks.
            (default: None)
        batch_id_fn (optional): if given, this callable object is
            used to extract the batch ids to be used in a possible exception.
            (default: None)
        progress_bar (optional): if ``True``, :mod:`tqdm` will be
            used to show a progress bar for each epoch. If a string is given,
            the content of the string will be shown before the progress bar.
            (default: None)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        data_loader: Optional[Iterable] = None,
        batch_input_fn: Optional[Callable] = None,
        batch_target_fn: Optional[Callable] = None,
        batch_id_fn: Optional[Callable] = None,
        progress_bar: Optional[Union[bool, str]] = None,
    ) -> None:
        self._model = model
        self._data_loader = data_loader
        self._batch_input_fn = batch_input_fn
        self._batch_target_fn = batch_target_fn
        self._batch_id_fn = batch_id_fn
        self._progress_bar = progress_bar

        self._epochs = 0
        self._iterations = 0
        self._must_stop = False
        self._hooks = {EPOCH_START: [], EPOCH_END: [], ITER_START: [], ITER_END: []}

    @property
    def batch_input_fn(self):
        return self._batch_input_fn

    @property
    def batch_target_fn(self):
        return self._batch_target_fn

    @property
    def batch_id_fn(self):
        return self._batch_id_fn

    @property
    def model(self):
        return self._model

    @property
    def hooks(self):
        return self._hooks

    def epochs(self):
        return self._epochs

    def iterations(self):
        return self._iterations

    @property
    def logger(self):
        return _logger

    @action
    def reset(self):
        """Reset the number of epochs and iterations run."""
        self._epochs = 0
        self._iterations = 0
        self._must_stop = False

    @action
    def stop(self):
        self._must_stop = True

    def set_data_loader(self, data_loader: Iterable):
        """Set the data loader object from which samples are loaded."""
        assert data_loader is not None
        self._data_loader = data_loader
        return self

    def set_batch_input_fn(self, fn: Optional[Callable]):
        """Set the function to obtain the inputs for the model.

        The argument can be either a function or a callable object that
        will receive as a single argument the batch read from the
        ``data_loader``, and must return the appropriate input for the
        model.
        """
        assert fn is None or callable(fn)
        self._batch_input_fn = fn
        return self

    def set_batch_target_fn(self, fn: Optional[Callable]):
        """Set the function to obtain the targets from the batch.

        The argument can be either a function or a callable object that
        will receive as a single argument the batch read from the
        ``data_loader``, it must return an appropriate object that the
        hooks can understand.
        """
        assert fn is None or callable(fn)
        self._batch_target_fn = fn
        return self

    def set_progress_bar(self, progress_bar: Union[bool, str]):
        self._progress_bar = progress_bar
        return self

    def add_hook(self, when: str, hook: laia.hooks.Hook):
        """Add a hook to be executed at some point during the run.

        When multiple hooks are added at the same point of the run, they will
        be run in order of addition.

        Args:
          when: point in the run (valid values: ``ITER_START``,
            ``ITER_END``, ``EPOCH_START``, ``EPOCH_END``).
          hook: `Hook` object.
        """
        assert when in self._hooks, "{!r} is not a valid hook event".format(when)
        self._hooks[when].append(hook)
        return self

    @action
    def run(self):
        """Run a single epoch on the `dataset_loader`."""
        assert self._data_loader is not None, "A data loader must be set"
        self._run_epoch()
        return self

    def _call_hooks(self, when: str, *args: Any, **kwargs: Any) -> None:
        for hook in self._hooks[when]:
            hook(*args, caller=self, **kwargs)

    def _run_iteration(self, batch_n: int, batch: Any) -> None:
        batch_input, batch_target = self._prepare_input_and_target(batch)

        action_kwargs = {
            "batch": batch,
            "batch_num": batch_n,
            "epoch": self._epochs,
            "iteration": self._iterations,
            "batch_input": batch_input,
            "batch_target": batch_target,
        }
        self._call_hooks(ITER_START, **action_kwargs)

        if self._must_stop:
            return

        # Put model in evaluation mode
        if hasattr(self._model, "eval"):
            self._model.eval()

        # Run model
        with self.exception_catcher(batch):
            batch_output = self._model(batch_input)

        self._iterations += 1
        action_kwargs["iteration"] = self._iterations
        action_kwargs["batch_output"] = batch_output
        self._call_hooks(ITER_END, **action_kwargs)

    def _prepare_input_and_target(self, batch: Any) -> (Any, Any):
        # Prepare input to the model.
        batch_input = self._batch_input_fn(batch) if self._batch_input_fn else batch
        # Prepare target to be passed to the loss function.
        batch_target = self._batch_target_fn(batch) if self.batch_target_fn else None
        return batch_input, batch_target

    def _run_epoch(self) -> None:
        self._call_hooks(EPOCH_START, epoch=self._epochs)

        if self._must_stop:
            return

        if self._progress_bar:
            batch_iterator = tqdm(
                self._data_loader,
                desc=self._progress_bar
                if isinstance(self._progress_bar, str)
                else None,
            )
        else:
            batch_iterator = self._data_loader

        for it, batch in enumerate(batch_iterator, 1):
            if self._must_stop:
                break
            self._run_iteration(it, batch)
        else:
            self._epochs += 1
            self._call_hooks(EPOCH_END, epoch=self._epochs)

    @contextmanager
    def exception_catcher(self, batch: Any) -> None:
        try:
            yield
        except Exception as e:
            raise EngineException(
                epoch=self._epochs,
                iteration=self._iterations,
                batch=self.batch_id_fn(batch) if self.batch_id_fn else batch,
                cause=e,
            ) from e

    def state_dict(self) -> Dict:
        return {
            "model": self._model.state_dict(),
            "epochs": self._epochs,
            "iterations": self._iterations,
            "hooks": {
                when: [
                    hook.state_dict() if hasattr(hook, "state_dict") else None
                    for hook in hooks
                ]
                for when, hooks in self._hooks.items()
            },
        }

    def load_state_dict(self, state: Dict) -> None:
        self._model.load_state_dict(state["model"])
        self._epochs = state["epochs"]
        self._iterations = state["iterations"]
        "Note: The hooks must be in the same order as those in the saved state"
        for when, hooks in self._hooks.items():
            hook_states = state["hooks"][when]
            for i, hook in enumerate(hooks):
                if i >= len(hook_states):
                    break
                if hasattr(hook, "load_state_dict"):
                    hook.load_state_dict(hook_states[i])

    @staticmethod
    def get_model_state_dict(state: Dict) -> Dict:
        return state["model"]


# If we decide to extend the Evaluator class, we can move it to
# a different file and extend from the Engine class, but
# right now an Evaluator is just a Engine.
Evaluator = Engine
