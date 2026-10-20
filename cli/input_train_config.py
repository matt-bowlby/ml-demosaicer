import copy

from train_config import TrainConfig, SingleImageConfig

from .menu import menu


def input_train_config(default_config: TrainConfig) -> TrainConfig:
    cfg = copy.deepcopy(default_config)
    while True:
        choice = menu(
            "Configure training",
            [
                "Done",
                f"Image width: {cfg.image_width}",
                f"Image height: {cfg.image_height}",
                f"Training seed range start: {cfg.seed_range_start}",
                f"Training seed range end: {cfg.seed_range_end}",
                f"Learning rate: {cfg.learning_rate}",
                f"Batch size: {cfg.batch_size}",
                f"Batch count: {cfg.batch_num}",
                f"Epoch count: {cfg.epoch_num}",
                f"Image generation version: {cfg.image_generation_version}",
                f"Weight seed: {cfg.weight_seed}",
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
                cfg.learning_rate = _float()
            case 7:
                cfg.batch_size = _integer()
            case 8:
                cfg.batch_num = _integer()
            case 9:
                cfg.epoch_num = _integer()
            case 10:
                cfg.image_generation_version = _integer()
            case 11:
                seed_choice = menu(
                    "Configure weight seed", ["Enter a seed", "Use a random seed"]
                )
                cfg.weight_seed = _integer() if seed_choice == 1 else ""
            case 12:
                cfg.single_image = menu("Train on a single image?", ["Yes", "No"]) == 1
                if cfg.single_image:
                    cfg.single_image_config = input_single_image_config(
                        cfg.single_image_config
                    )


def input_single_image_config(
    default_single_image_config: SingleImageConfig,
) -> SingleImageConfig:
    cfg: SingleImageConfig = copy.deepcopy(default_single_image_config)
    while True:
        choice = menu(
            "Configure single image options",
            [
                "Done",
                f"Image seed: {cfg.image_seed}",
            ],
        )

        match choice:
            case 1:
                return cfg
            case 2:
                cfg.image_seed = _integer()


def _integer(prompt: str | None = "Enter Value") -> int:
    if prompt is not None:
        print(prompt)
    while True:
        try:
            return int(input("> "))
        except:
            continue


def _float(prompt: str | None = "Enter Value") -> float:
    if prompt is not None:
        print(prompt)
    while True:
        try:
            return float(input("> "))
        except:
            continue
