"""Device sampling plus a separate CPU stream for Python control flow."""

from dataclasses import dataclass
import torch

from device import device


@dataclass
class RandomStreams:
    device_rng: torch.Generator
    control_rng: torch.Generator

    @classmethod
    def from_seed(cls, seed: int | None = None) -> "RandomStreams":
        """Create both streams from one seed; None chooses a fresh seed."""
        rng = torch.Generator(device=device)
        if seed is None:
            rng.seed()
        else:
            rng.manual_seed(seed)
        return cls.from_generator(rng)

    @classmethod
    def from_generator(cls, rng: torch.Generator) -> "RandomStreams":
        # Consume the supplied stream so repeated images advance, and reseeding
        # reproduces both streams. This is one scalar read per image on a GPU.
        seed = int(torch.randint(0, 2**63 - 1, (), generator=rng, device=rng.device).item())
        return cls(rng, torch.Generator(device="cpu").manual_seed(seed))



def rngr_control(rng: RandomStreams) -> float:
    return float(torch.rand((), generator=rng.control_rng, device="cpu").item())


def rngd_control(rng: RandomStreams, peak: float = 0.15, concentration: float = 2.2) -> float:
    generator = rng.control_rng
    a = torch.tensor(1 + peak * concentration, dtype=torch.float32, device="cpu")
    b = torch.tensor(1 + (1 - peak) * concentration, dtype=torch.float32, device="cpu")
    x = torch._standard_gamma(a, generator=generator)
    y = torch._standard_gamma(b, generator=generator)
    return float((x / (x + y)).item())


def rngr_control_shape(rng: RandomStreams, shape) -> torch.Tensor:
    return torch.rand(shape, dtype=torch.float32, generator=rng.control_rng, device="cpu")


def rngr(rng: RandomStreams) -> torch.Tensor:
    return torch.rand((), generator=rng.device_rng, device=device)


def rngi(rng: RandomStreams, low: int, high: int | None = None) -> int:
    if high is None:
        low, high = 0, low
    return int(torch.randint(low, high, (), generator=rng.control_rng, device="cpu").item())


def rngr_shape(rng: RandomStreams, shape) -> torch.Tensor:
    if isinstance(shape, int):
        shape = (shape,)
    return torch.rand(shape, dtype=torch.float32, generator=rng.device_rng, device=device)


def rngd_shape(rng: RandomStreams, shape, peak: float = 0.15, concentration: float = 2.2) -> torch.Tensor:
    if isinstance(shape, int):
        shape = (shape,)
    # Beta.sample does not accept a generator. A beta variate is a ratio of
    # independent gamma variates; this primitive supports explicit RNG state.
    a = torch.full(shape, 1 + peak * concentration, dtype=torch.float32, device=device)
    b = torch.full(shape, 1 + (1 - peak) * concentration, dtype=torch.float32, device=device)
    x = torch._standard_gamma(a, generator=rng.device_rng)
    y = torch._standard_gamma(b, generator=rng.device_rng)
    return x / (x + y)


def rngd(rng: RandomStreams, peak: float = 0.15, concentration: float = 2.2) -> torch.Tensor:
    return rngd_shape(rng, (), peak, concentration)
