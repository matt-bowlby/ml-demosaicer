from pathlib import Path

from .new_training_run import new_training_run
from .continue_previous_run import continue_previous_run
from .input_train_config import input_train_config
from .menu import menu
from .cli_config import CliConfig
from .input_eval_config import input_eval_config
from .eval_model import eval_model

def cli():
    while True:
        print("Welcome to the ML Demosaicer CLI.")

        cli_config = CliConfig.load("./cli_config.json")
        if cli_config is None:
            cli_config = CliConfig()
            print(
                "To begin, enter the path of the folder where you want to store all of the training runs:"
            )
            path = None
            while True:
                try:
                    path = Path(input("> ")).absolute()
                except:
                    continue
                if not path.exists() or not path.is_dir():
                    continue

                cli_config.training_run_destination = str(path)
                cli_config.save("./cli_config.json")
                break

        while True:
            option = menu(
                "What would you like to do?",
                [
                    "Start a new training run",
                    "Continue a previous training run",
                    "Evaluate a previous training run",
                    "Edit the default train configuration",
                    "Edit the default eval configuration"
                ],
            )

            match option:
                case 1:
                    new_training_run(cli_config)
                case 2:
                    continue_previous_run(cli_config)
                case 3:
                    eval_model(cli_config)
                case 4:
                    cli_config.default_train_config = input_train_config(cli_config.default_train_config)
                    cli_config.save("./cli_config.json")
                case 5:
                    cli_config.default_eval_config = input_eval_config(cli_config.default_eval_config)
                    cli_config.save("./cli_config.json")
