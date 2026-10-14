from pathlib import Path
import json

from train_config import TrainConfig
from eval_config import EvalConfig


class CliConfig:
    def __init__(self) -> None:
        self.training_run_destination: str | None = None
        self.default_train_config: TrainConfig = TrainConfig()
        self.default_eval_config: EvalConfig = EvalConfig()

    @staticmethod
    def load(path: str) -> CliConfig | None:
        try:
            path_obj = Path(path)
        except:
            return None
        if not path_obj.exists() or not path_obj.is_file():
            return None

        with open(path_obj, "r", encoding="utf-8") as f:
            info = json.load(f)

        config = CliConfig()
        config.training_run_destination = info.get("training_run_destination")
        config.default_train_config = TrainConfig(info.get("default_train_config"))
        config.default_eval_config = EvalConfig(info.get("default_eval_config"))

        return config

    def save(self, path: str):
        path_obj = Path(path)
        with open(path_obj, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent="\t")

    def to_dict(self):
        return {
            "training_run_destination": self.training_run_destination,
            "default_train_config": self.default_train_config.to_dict(),
            "default_eval_config": self.default_eval_config.to_dict(),
        }