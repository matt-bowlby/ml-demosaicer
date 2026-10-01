import json

class SeedRun:
    def __init__(self, run_object):
        self.seed = run_object.seed
        self.images_generated = run_object.images_generated
        self.aborted = run_object.aborted

class TrainConfig:
    def __init__(self, config_path: str, runs_path: str):
        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file)

            self.image_width = config.get("image_width")
            self.image_height = config.get("image_height")
            self.training_range_start = config.get("training_range_start")
            self.training_range_end = config.get("training_range_end")
            self.testing_range_start = config.get("testing_range_start")
            self.testing_range_end = config.get("testing_range_end")
            self.learning_rate = config.get("learning_rate")
            self.image_generation_version = config.get("image_generation_version")

        with open(runs_path, "r", encoding="utf-8") as file:
            runs = json.load(file)
            self.runs = [SeedRun(run_object) for run_object in runs]