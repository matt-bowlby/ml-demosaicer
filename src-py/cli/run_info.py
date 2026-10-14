from pathlib import Path
import json
from train_config import TrainConfig


class RunInfo:
    def __init__(self, train_config: TrainConfig, starting_offset: int = 0) -> None:
        self.train_config = train_config
        self.image_offset = starting_offset

    @staticmethod
    def load_infos(path: str) -> list["RunInfo"] | None:
        try:
            path_obj = Path(path)
        except:
            return None
        if not path_obj.exists() or not path_obj.is_file():
            return None

        with open(path_obj, "r", encoding="utf-8") as f:
            info = json.load(f)

        infos = []
        for i in info:
            infos.append(
                RunInfo(TrainConfig(i.get("train_config")), i.get("image_offset", 0))
            )

        return infos

    def save(self, path: str):
        path_obj = Path(path)
        infos = []
        if path_obj.exists():
            with open(path_obj, "r", encoding="utf-8") as f:
                try:
                    infos = json.load(f)
                except:
                    pass

        infos.append(self.to_dict())

        with open(path_obj, "w+", encoding="utf-8") as f:
            json.dump(infos, f, indent="\t")

    def to_dict(self):
        return {
            "train_config": self.train_config.to_dict(),
            "image_offset": self.image_offset,
        }
