from line_profiler import LineProfiler
import pstats
import numpy as np

from generate_image.v1 import fills

profiler = LineProfiler()

def benchmark():
    for seed in range(30):
        rng = np.random.default_rng(seed)
        fills.noise(rng.random() * 0.5, np.array([0.5, 0.5, 0.5]), rng, 1000, 1000)

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