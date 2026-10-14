from torch.utils.data import IterableDataset
import numpy as np

from generate_image import generate_image, bayer_mosaic
from train_config import TrainConfig

from torch.utils.data import IterableDataset
import numpy as np


class TrainDataset(IterableDataset):
    def __init__(self, config: TrainConfig, offset: int = 0):
        self.config = config
        self.num_images = self.config.batch_num * self.config.batch_size
        self.single_image = config.single_image
        self.single_image_config = config.single_image_config
        self.offset = offset

    def __iter__(self):
        if self.single_image:
            for x, y, seed in self._single_image():
                yield (x, y, seed)
        else:
            for x, y, seed in self._random_image():
                yield x, y, seed

    def _random_image(self):
        # Seed range endpoints are inclusive.
        seed = self.config.seed_range_start + self.offset
        for _ in range(self.num_images):
            rng = np.random.default_rng(seed)

            if seed >= self.config.seed_range_end:
                return

            image = generate_image(
                rng,
                self.config.image_width,
                self.config.image_height,
                self.config.image_generation_version,
            )
            yield (bayer_mosaic(image, "RGGB"), image, seed)
            seed += 1

    def _single_image(self):
        seed = self.single_image_config.image_seed
        rng = np.random.default_rng(seed)
        image = generate_image(
            rng,
            self.config.image_width,
            self.config.image_height,
            self.config.image_generation_version,
        )
        for _ in range(self.num_images):
            yield (bayer_mosaic(image, "RGGB"), image, seed)