import json

class TrainConfig:
    runs: dict

    def __init__(self, config_path: str, runs_path: str):
        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file)

            self.config_path = config_path
            self.runs_path = runs_path

            self.image_width = config.get("image_width")
            self.image_height = config.get("image_height")
            self.training_range_start = config.get("training_range_start")
            self.training_range_end = config.get("training_range_end")
            self.testing_range_start = config.get("testing_range_start")
            self.testing_range_end = config.get("testing_range_end")
            self.max_generations_per_seed = config.get("max_generations_per_seed")
            self.learning_rate = config.get("learning_rate")
            self.batch_size = config.get("batch_size")
            self.batch_num = config.get("batch_num")
            self.image_generation_version = config.get("image_generation_version")
            self.weight_seed = config.get("weight_seed")
            self.training_seed = config.get("training_seed")
            self.single_image = config.get("single_image")


        with open(runs_path, "r", encoding="utf-8") as file:
            runs = json.load(file)
            self.runs = runs

    def update_run(
        self,
        seed: int,
        images_generated: int | None = None,
        aborted: bool | None = None,
        version: int | None = None
    ):
        str_seed = str(seed)
        entry = self.runs.get(seed, None)
        if images_generated is None:
            images_generated = 0 if entry is None else entry.get(str_seed).get("images_generated")
        if aborted is None:
            aborted = False if entry is None else entry.get(str_seed).get("aborted")
        if version is None:
            version = 1 if entry is None else entry.get(str_seed).get("version")

        self.runs[str_seed] = {
            "images_generated": images_generated,
            "aborted": aborted,
            "version": version
        }

    def save_runs(self):
        with open(self.runs_path, "w", encoding="utf-8") as file:
            json.dump(self.runs, file, indent=2)
