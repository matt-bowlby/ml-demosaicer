import json


class SingleImageConfig:
    def __init__(self, info: dict | None = None) -> None:
        info = {k: v for k, v in (info or {}).items() if v is not None}
        self.image_seed: int = info.get("image_seed", 1000000565)

    @staticmethod
    def from_str(encoded: str) -> "SingleImageConfig":
        decoded = json.loads(encoded)
        if not isinstance(decoded, dict):
            raise ValueError("Training config must be a JSON object")
        return SingleImageConfig(decoded)

    def to_dict(self):
        return {"image_seed": self.image_seed}


class TrainConfig:
    def __init__(self, info: dict | None = None) -> None:
        info = {k: v for k, v in (info or {}).items() if v is not None}
        self.image_width: int = info.get("image_width", 32)
        self.image_height: int = info.get("image_height", 32)
        self.seed_range_start: int = info.get("seed_range_start", 0)
        self.seed_range_end: int = info.get("seed_range_end", 999999999)
        self.training_seed: int = info.get("training_seed", 42)
        self.learning_rate: float = info.get("learning_rate", 0.0005)
        self.batch_size: int = info.get("batch_size", 5)
        self.batch_num: int = info.get("batch_num", 20000)
        self.epoch_num: int = info.get("epoch_num", 1)
        self.image_generation_version: int = info.get("image_generation_version", 1)
        self.weight_seed: int | str = info.get("weight_seed", "")
        self.single_image: bool = info.get("single_image", False)
        self.single_image_config: SingleImageConfig = SingleImageConfig(
            info.get("single_image_config", {"image_seed": 1000000565})
        )

    @staticmethod
    def from_str(encoded: str) -> "TrainConfig":
        decoded = json.loads(encoded)
        if not isinstance(decoded, dict):
            raise ValueError("Training config must be a JSON object")
        return TrainConfig(decoded)

    def to_dict(self):
        return {
            "image_width": self.image_width,
            "image_height": self.image_height,
            "seed_range_start": self.seed_range_start,
            "seed_range_end": self.seed_range_end,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "batch_num": self.batch_num,
            "epoch_num": self.epoch_num,
            "image_generation_version": self.image_generation_version,
            "weight_seed": self.weight_seed,
            "single_image": self.single_image,
            "single_image_config": self.single_image_config.to_dict(),
        }
