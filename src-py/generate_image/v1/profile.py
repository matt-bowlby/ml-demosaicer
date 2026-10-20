from generate_image import RandomStreams
import torch
from line_profiler import LineProfiler
import numpy as np

from generate_image.v1 import fills
from device import device

profiler = LineProfiler()

def benchmark():
    for seed in range(30):
        rng = RandomStreams.from_seed(seed)
        fills.noise(torch.rand((), generator=rng.device_rng, device=device) * 0.5, torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device=device), rng, 1000, 1000)

for function in (
    fills.noise,
    fills.checkerboard,
    fills.radial_gradient,
    fills.stripes,
    fills._blend,
):
    profiler.add_callable(function)


profiler(benchmark)()

profiler.print_stats(output_unit=1e-3)