import copy

from eval_config import EvalConfig

from .menu import menu
from .input_train_config import input_single_image_config


def input_eval_config(default_config: EvalConfig) -> EvalConfig:
    cfg = copy.deepcopy(default_config)
    while True:
        choice = menu(
            "Configure evaluation",
            [
                "Done",
                f"Image width: {cfg.image_width}",
                f"Image height: {cfg.image_height}",
                f"Evaluation seed range start: {cfg.seed_range_start}",
                f"Evaluation seed range end: {cfg.seed_range_end}",
                f"Number of images generated per seed: {cfg.images_per_seed}",
                f"Number of images in evaluation set: {cfg.num_images}",
                f"Single image: {cfg.single_image}",
            ],
        )

        match choice:
            case 1:
                return cfg
            case 2:
                cfg.image_width = _integer()
            case 3:
                cfg.image_height = _integer()
            case 4:
                cfg.seed_range_start = _integer()
            case 5:
                cfg.seed_range_end = _integer()
            case 6:
                cfg.images_per_seed = _integer()
            case 7:
                cfg.num_images = _integer()
            case 8:
                cfg.single_image = menu("Train on a single image?", ["Yes", "No"]) == 1
                if cfg.single_image:
                    cfg.single_image_config = input_single_image_config(cfg.single_image_config)


def _integer(prompt: str | None = "Enter Value") -> int:
    if prompt is not None:
        print(prompt)
    while True:
        try:
            return int(input("> "))
        except:
            continue