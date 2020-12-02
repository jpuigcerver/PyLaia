from laia.dummies import DummyModel
from laia.engine import EvaluatorModule


class DummyEvaluator(EvaluatorModule):
    def __init__(self):
        super().__init__(
            model=DummyModel((3, 3), 10, horizontal=True),
            batch_input_fn=self.batch_input_fn,
            batch_id_fn=self.batch_id_fn,
        )

    @staticmethod
    def batch_input_fn(batch):
        return batch["img"]

    @staticmethod
    def batch_id_fn(batch):
        return batch["id"]
