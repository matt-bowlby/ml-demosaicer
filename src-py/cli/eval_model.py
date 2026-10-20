from pathlib import Path
import torch

from learning import evaluate, predict, create_eval_data_loader, Demosaicer, TrainResult
from generate_image import RandomStreams, generate_image, bayer_mosaic
from device import device

from .cli_config import CliConfig
from .menu import menu
from .input_eval_config import input_eval_config

def eval_model(cli_config: CliConfig):
    while True:
        print("Enter the name of the run you want to evaluate:")
        name = input("> ")

        if cli_config.training_run_destination is None:
            raise ValueError("Training run destination is not given")

        try:
            model_path = Path(cli_config.training_run_destination, f"{name}.pt")
            if not model_path.exists():
                print("Training run does not exist.")
                continue

            info_path = Path(cli_config.training_run_destination, f"{name}_info.json")
            if not info_path.exists():
                print("Info does not exist.")
                continue

            result_path = Path(
                cli_config.training_run_destination, f"{name}_training_results.json"
            )
            if not result_path.exists():
                print("Result does not exist.")
                continue
        except:
            continue

        break

    weights = torch.load(model_path, map_location="cpu", weights_only=True)
    model = Demosaicer().to(device)
    model.load_state_dict(weights)

    config = input_eval_config(cli_config.default_eval_config)
    result = evaluate(model, create_eval_data_loader(config))

    evaluation_choice = menu(
        "How would you like to evaluate the dataset?",
        ["Show statistics", "Random samples", "None"],
    )
    match evaluation_choice:
        case 1:
            train = TrainResult.load(str(result_path))
            result.add_train_result(train)
            result.visualize()
        case 2:
            while True:
                rng = RandomStreams.from_seed()
                image = generate_image(rng, config.image_width, config.image_height, config.image_generation_version)
                mosaic = bayer_mosaic(image, "RGGB")
                predict(model, mosaic, image)
                print("Enter q to quit, or anything else to show another image:")
                answer = input("> ").strip().lower()
                if answer == "q":
                    break
        case 3:
            pass