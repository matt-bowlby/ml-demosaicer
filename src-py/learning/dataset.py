from typing import Any

from torch.utils.data import IterableDataset
import numpy as np

from generate_image import generate_image, bayer_mosaic
from config import TrainConfig

from torch.utils.data import IterableDataset
import numpy as np


class RawDataset(IterableDataset):
    def __init__(self, config: TrainConfig):
        self.config = config
        self.num_samples = config.batch_num * config.batch_size

    def _seed_range(self):
        return range(
            self.config.training_range_start,
            self.config.training_range_end + 1,
        )

    def __iter__(self):
        samples = 0

        # Seed range endpoints are inclusive.
        for seed in self._seed_range():
            rng = np.random.default_rng(seed)

            for _ in range(self.config.max_generations_per_seed):
                if samples >= self.num_samples:
                    return

                image = generate_image(
                    rng,
                    self.config.image_width,
                    self.config.image_height,
                    self.config.image_generation_version,
                )

                samples += 1
                yield (bayer_mosaic(image, "RGGB"), image, seed)


class EvalDataset(RawDataset):
    """Generate evaluation samples using the configured testing seed range."""

    def __init__(self, config: TrainConfig):
        super().__init__(config)
        self.num_samples = config.eval_num

    def _seed_range(self):
        return range(
            self.config.testing_range_start,
            self.config.testing_range_end + 1,
        )


class SingleImageDataset(IterableDataset):
    def __init__(self, config: TrainConfig):
        self.config = config
        self.num_samples = config.batch_num * config.batch_size
        self.seed = (
            config.training_seed
            if config.training_seed is not None
            else config.testing_range_start
        )
        rng = np.random.default_rng(self.seed)
        self.img = generate_image(
            rng,
            self.config.image_width,
            self.config.image_height,
            self.config.image_generation_version,
        )
        self.mosaic = bayer_mosaic(self.img, "RGGB")

    def __iter__(self):
        for i in range(self.num_samples):
            yield (self.mosaic, self.img, self.seed)
