"""Aspect-preserving full-image preparation for the optional neural model."""
from dataclasses import dataclass
import math

from PIL import Image
import torch
from torch.nn import functional as F

from .state import check


@dataclass(frozen=True)
class PreparedImage:
    pixels: torch.Tensor
    valid: torch.Tensor
    source_size: tuple[int, int]
    resized_size: tuple[int, int]
    scale_x: float
    scale_y: float

    @property
    def padded_size(self):
        return self.pixels.shape[-1], self.pixels.shape[-2]


def prepare_image(image: Image.Image, *, max_side=960, max_source_pixels=8_000_000):
    check(type(max_side) is int and 64 <= max_side <= 1536, "max side must be 64..1536")
    width, height = image.size
    check(width > 0 and height > 0 and width*height <= max_source_pixels, "source pixel budget exceeded")
    check(max(width, height) <= 16384, "source dimension budget exceeded")
    scale = min(1., max_side/max(width, height))
    size = (max(1, round(width*scale)), max(1, round(height*scale)))
    # The actual rounded x/y ratios, not the nominal resize factor, own geometry.
    rgba = image.convert("RGBA").resize(size, Image.Resampling.LANCZOS)
    # Composite onto a declared black canvas; discard no alpha by simply
    # revealing RGB stored in fully transparent pixels. Those pixels are invalid.
    canvas = Image.new("RGBA", size, (0, 0, 0, 255))
    canvas.alpha_composite(rgba)
    resized = canvas.convert("RGB")
    pixels = torch.frombuffer(bytearray(resized.tobytes()), dtype=torch.uint8)
    pixels = pixels.reshape(size[1], size[0], 3).permute(2, 0, 1).float().div_(255)
    padded_width, padded_height = math.ceil(size[0]/16)*16, math.ceil(size[1]/16)*16
    pixels = F.pad(pixels, (0, padded_width-size[0], 0, padded_height-size[1])).contiguous()
    valid = torch.zeros((1, padded_height, padded_width), dtype=torch.bool)
    alpha = torch.frombuffer(bytearray(rgba.getchannel("A").tobytes()), dtype=torch.uint8).reshape(size[1], size[0])
    valid[:, :size[1], :size[0]] = alpha > 0
    return PreparedImage(pixels, valid, (width, height), size, size[0]/width, size[1]/height)
