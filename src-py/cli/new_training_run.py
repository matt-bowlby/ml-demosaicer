from pathlib import Path
import time
import json

import numpy as np
import torch

from learning import train_default, create_training_data_loader

from .input_train_config import input_train_config
from .print_progress import print_progress
from .cli_config import CliConfig
from.run_info import RunInfo
from .play_sound import play_sound


def new_training_run(cli_config: CliConfig):
    while True:
        print("Enter the name for this training run:")
        name = input("> ")

        if cli_config.training_run_destination is None:
            raise ValueError("Training run destination is not given")

        try:
            model_path = Path(cli_config.training_run_destination, f"{name}.pt")
            if model_path.exists():
                print("Training run name already exists.")
                continue

            info_path = Path(cli_config.training_run_destination, f"{name}_info.json")
            if info_path.exists():
                print("Training run info name already exists.")
                continue

            result_path = Path(
                cli_config.training_run_destination, f"{name}_training_results.json"
            )
        except:
            continue

        break

    config = input_train_config(cli_config.default_train_config)

    print("Training...")

    data_loader = create_training_data_loader(config)
    batch_index = 1
    model = None
    result = None
    WINDOW = 1000
    last_times = []
    start = time.perf_counter()
    print()
    for m, r, _ in train_default(data_loader, config):
        elapsed = time.perf_counter() - start
        last_times.append(elapsed)
        if len(last_times) > WINDOW:
            last_times.pop(0)
        avg_time = np.array(last_times).sum() / float(len(last_times))
        remaining_time = avg_time * (config.batch_num - batch_index)
        hours = remaining_time // 3600
        minutes = (remaining_time % 3600) // 60
        seconds = remaining_time % 60
        print(
            f"\033[1A\r\033[2KEstimated remaining time: {hours}h {minutes}m {seconds}s",
            flush=True,
        )
        print_progress(batch_index, config.batch_num)
        batch_index += 1
        model = m
        result = r

        start = time.perf_counter()

    if model is None or result is None:
        raise ValueError("Model is None")

    torch.save(model.state_dict(), model_path)
    info = RunInfo(config, config.batch_size * config.batch_num)
    info.save(str(info_path))
    with open(result_path, "w+") as f:
        json.dump(result.to_dict(), f)
    play_sound()

    print()
