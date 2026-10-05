import cProfile
import pstats

import numpy as np
from generate_image import generate_image

def benchmark():
    for seed in range(20):
        generate_image(np.random.default_rng(seed), 6100, 4100, 1)

profiler = cProfile.Profile()
profiler.runcall(benchmark)

pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(20)