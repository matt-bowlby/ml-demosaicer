import torch
import cProfile
import pstats

from generate_image import RandomStreams, generate_image

def benchmark():
    for seed in range(20):
        generate_image(RandomStreams.from_seed(seed), 6100, 4100, 1)
        print("Generated image")

profiler = cProfile.Profile()
profiler.runcall(benchmark)

pstats.Stats(profiler).strip_dirs().sort_stats("tottime").print_stats(20)