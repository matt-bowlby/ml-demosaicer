import json

from train_config import SingleImageConfig


class EvalConfig:
    def __init__(self, info: dict | None = None) -> None:
        info = {k: v for k, v in (info or {}).items() if v is not None}
        self.image_width: int = info.get("image_width", 32)
        self.image_height: int = info.get("image_height", 32)
        self.seed_range_start: int = info.get("seed_range_start", 1000000000)
        self.seed_range_end: int = info.get("seed_range_end", 1999999999)
        self.images_per_seed: int = info.get("images_per_seed", 1)
        self.num_images: int = info.get("num_images", 1)
        self.image_generation_version = info.get("image_generation_version", 1)
        self.single_image: bool = info.get("single_image", False)
        self.single_image_config: SingleImageConfig = SingleImageConfig(
            info.get("single_image_config", {"image_seed": 1000000565})
        )

    @staticmethod
    def from_str(encoded: str) -> "EvalConfig":
        decoded = json.loads(encoded)
        if not isinstance(decoded, dict):
            raise ValueError("Evaluation config must be a JSON object")
        return EvalConfig(decoded)

    def to_dict(self):
        return {
            "image_width": self.image_width,
            "image_height": self.image_height,
            "seed_range_start": self.seed_range_start,
            "seed_range_end": self.seed_range_end,
            "images_per_seed": self.images_per_seed,
            "num_images": self.num_images,
            "image_generation_version": self.image_generation_version,
            "single_image": self.single_image,
            "single_image_config": self.single_image_config.to_dict(),
        }
