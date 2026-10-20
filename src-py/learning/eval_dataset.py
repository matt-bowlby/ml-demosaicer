import torch
from torch.utils.data import IterableDataset

from generate_image import RandomStreams, generate_image, bayer_mosaic
from eval_config import EvalConfig
from device import device


class EvalDataset(IterableDataset):
    def __init__(self, config: EvalConfig, offset: int = 0):
        self.config = config
        self.num_images = self.config.num_images
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
            rng = RandomStreams.from_seed(seed)

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
        rng = RandomStreams.from_seed(seed)
        image = generate_image(
            rng,
            self.config.image_width,
            self.config.image_height,
            self.config.image_generation_version,
        )
        for _ in range(self.num_images):
            yield (bayer_mosaic(image, "RGGB"), image, seed)