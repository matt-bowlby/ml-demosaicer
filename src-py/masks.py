from typing import Literal
import torch

from device import device

MaskType = Literal["RGGB", "BGGR", "GRBG", "GBRG", "X-Trans"]

_PATTERNS = {
    "RGGB": ("RG", "GB"),
    "BGGR": ("BG", "GR"),
    "GRBG": ("GR", "BG"),
    "GBRG": ("GB", "RG"),
    "X-Trans": ("GBGGRG", "RGRBGB", "GBGGRG", "GRGGBG", "BGBRGR", "GRGGBG"),
}

def mask_image(
    image: torch.Tensor, channel: Literal["r", "g", "b"], masktype: MaskType
) -> torch.Tensor:
    """Keep sampled pixels of a channel, preserving any batch dimension.

    Accept (3, height, width) or (N, 3, height, width) and return
    (1, height, width) or (N, 1, height, width), respectively.

    Other pixels are zero. Preserve the input dtype without modifying the input.
    Pattern tiles (including the 6x6 X-Trans tile above) start at the top-left.
    """
    index = ("r", "g", "b").index(channel)
    source = torch.as_tensor(image)[..., index : index + 1, :, :]
    result = torch.zeros_like(source, dtype=torch.float32, device=device)
    pattern = _PATTERNS[masktype]
    tile_height, tile_width = len(pattern), len(pattern[0])
    for row, colors in enumerate(pattern):
        for col, color in enumerate(colors):
            if color == channel.upper():
                result[..., row::tile_height, col::tile_width] = source[
                    ..., row::tile_height, col::tile_width
                ]
    return result
