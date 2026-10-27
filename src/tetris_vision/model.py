"""Random-initialized M2/M3 lattice-query model; no detector/palette dependency.

This is the one-board, axis-aligned affine prototype. It deliberately exposes
raw heads; optional M3 components are experimental, with no calibrated parsing.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .state import check


@dataclass(frozen=True)
class ModelConfig:
    widths: tuple[int, int, int, int] = (48, 96, 192, 256)
    dimension: int = 192
    heads: int = 6
    blocks: int = 6
    refinements: int = 2
    visible_rows: int = 20
    spawn_rows: int = 6
    columns: int = 10
    max_batch: int = 4
    max_pixels: int = 1_048_576
    component_capacity: int = 0
    layer_evidence: bool = False

    def __post_init__(self):
        check(type(self.heads) is int and 1 <= self.heads <= 8, "invalid attention head count")
        check(len(self.widths) == 4 and all(type(c) is int and 8 <= c <= 256 and c % 8 == 0 for c in self.widths), "invalid encoder widths")
        check(16 <= self.dimension <= 256 and self.dimension % 4 == 0 and self.dimension % self.heads == 0, "invalid attention dimension")
        check(1 <= self.blocks <= 6 and 0 <= self.refinements <= min(3, self.blocks), "bounded decoder configuration required")
        check(self.visible_rows == 20 and self.spawn_rows == 6 and self.columns == 10, "prototype capacity is one 10x20 board plus six spawn rows")
        check(1 <= self.max_batch <= 4 and 4096 <= self.max_pixels <= 1536**2, "invalid work capacity")
        check(type(self.component_capacity) is int and 0 <= self.component_capacity <= 64,
              "component capacity must be0 (M2) or1..64 (M3)")
        check(type(self.layer_evidence) is bool, "layer evidence must be an explicit boolean")

    @classmethod
    def pilot(cls):
        return cls(widths=(16, 32, 64, 96), dimension=64, heads=4, blocks=2)


class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                                    nn.GroupNorm(8, channels), nn.SiLU(),
                                    nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                                    nn.GroupNorm(8, channels))

    def forward(self, pixels):
        return F.silu(pixels+self.layers(pixels))


class WindowContext(nn.Module):
    def __init__(self, dimension, window=8):
        super().__init__()
        self.window = window
        self.norm = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(dimension, 4, batch_first=True)

    def forward(self, features, valid):
        batch, channels, height, width = features.shape
        w = self.window
        ph, pw = math.ceil(height/w)*w, math.ceil(width/w)*w
        value = F.pad(features, (0, pw-width, 0, ph-height))
        mask = F.adaptive_max_pool2d(valid.float(), (height, width)).bool()
        mask = F.pad(mask, (0, pw-width, 0, ph-height))
        windows = value.permute(0, 2, 3, 1).reshape(batch, ph//w, w, pw//w, w, channels)
        windows = windows.permute(0, 1, 3, 2, 4, 5).reshape(-1, w*w, channels)
        observed = mask[:, 0].reshape(batch, ph//w, w, pw//w, w).permute(0, 1, 3, 2, 4).reshape(-1, w*w)
        safe = observed.clone()
        safe[~safe.any(dim=1), 0] = True  # Avoid all-masked softmax, then zero those queries.
        normalized = self.norm(windows)
        attended = self.attention(normalized, normalized, normalized, key_padding_mask=~safe, need_weights=False)[0]
        windows = (windows+attended)*observed.unsqueeze(-1)
        value = windows.reshape(batch, ph//w, pw//w, w, w, channels).permute(0, 1, 3, 2, 4, 5)
        return value.reshape(batch, ph, pw, channels).permute(0, 3, 1, 2)[:, :, :height, :width]


class ImageEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        stages, previous = [], 3
        for channels in config.widths:
            # Even kernel/stride gives feature centers at stride/2 in source
            # boundary coordinates, matching align_corners=False sampling.
            stages.append(nn.Sequential(nn.Conv2d(previous, channels, 4, stride=2, padding=1, bias=False),
                                        nn.GroupNorm(8, channels), nn.SiLU(), Residual(channels)))
            previous = channels
        self.stages = nn.ModuleList(stages)
        self.window = WindowContext(config.widths[2])
        self.detail2 = nn.Conv2d(config.widths[0], config.dimension, 1)
        self.detail4 = nn.Conv2d(config.widths[1], config.dimension, 1)
        self.context8 = nn.Conv2d(config.widths[2], config.dimension, 1)
        self.context16 = nn.Conv2d(config.widths[3], config.dimension, 1)

    def forward(self, pixels, valid):
        first = self.stages[0]((2*pixels-1)*valid)
        second = self.stages[1](first)
        third = self.window(self.stages[2](second), valid)
        fourth = self.stages[3](third)
        detail = self.detail4(second)+F.avg_pool2d(self.detail2(first), 2)
        detail = detail+F.interpolate(self.context8(third), size=second.shape[-2:], mode="bilinear", align_corners=False)
        coarse = self.context16(fourth)
        mask = F.adaptive_max_pool2d(valid.float(), coarse.shape[-2:])
        size = (min(16, coarse.shape[-2]), min(16, coarse.shape[-1]))
        weight = F.adaptive_avg_pool2d(mask, size)
        context = F.adaptive_avg_pool2d(coarse*mask, size)/weight.clamp_min(1e-6)
        return detail, context, weight[:, 0] > 0


def spatial_encoding(height, width, dimension, device, dtype):
    y, x = torch.meshgrid(torch.arange(height, device=device, dtype=dtype)+.5,
                          torch.arange(width, device=device, dtype=dtype)+.5, indexing="ij")
    frequency = 2*math.pi*torch.arange(1, dimension//4+1, device=device, dtype=dtype)
    return torch.cat(((x[..., None]/width*frequency).sin(), (x[..., None]/width*frequency).cos(),
                      (y[..., None]/height*frequency).sin(), (y[..., None]/height*frequency).cos()), dim=-1).reshape(height*width, dimension)


def decode_lattice(raw, columns=10, rows=20):
    """Nonfolding axis-aligned affine basis in normalized image boundaries."""
    values = raw.sigmoid()
    left, top = values[:, 0]*.8, values[:, 1]*.8
    width, height = (1-left)*(.05+.95*values[:, 2]), (1-top)*(.05+.95*values[:, 3])
    zero = torch.zeros_like(left)
    return torch.stack((torch.stack((left, top+height), -1), torch.stack((width/columns, zero), -1),
                        torch.stack((zero, -height/rows), -1)), dim=1)


def lattice_points(lattice, coordinates, offsets):
    locations = coordinates[None, :, None, :]+offsets
    return (lattice[:, None, None, 0]+locations[..., :1]*lattice[:, None, None, 1]
            +locations[..., 1:]*lattice[:, None, None, 2])


def sample_features(features, lattice, coordinates, offsets):
    points = lattice_points(lattice, coordinates, offsets)
    samples = F.grid_sample(features, 2*points-1, padding_mode="zeros", align_corners=False)
    return samples.permute(0, 2, 3, 1), points


class AxialDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        dimension, heads = config.dimension, config.heads
        self.row = nn.MultiheadAttention(dimension, heads, batch_first=True)
        self.column = nn.MultiheadAttention(dimension, heads, batch_first=True)
        self.context = nn.MultiheadAttention(dimension, heads, batch_first=True)
        self.norms = nn.ModuleList(nn.LayerNorm(dimension) for _ in range(4))
        self.ff = nn.Sequential(nn.Linear(dimension, 2*dimension), nn.GELU(), nn.Linear(2*dimension, dimension))
        self.rows = config.visible_rows+config.spawn_rows
        self.columns = config.columns

    def forward(self, cells, context, context_valid):
        batch, _, dimension = cells.shape
        value = self.norms[0](cells).reshape(batch*self.rows, self.columns, dimension)
        cells = cells+self.row(value, value, value, need_weights=False)[0].reshape_as(cells)
        value = self.norms[1](cells).reshape(batch, self.rows, self.columns, dimension).transpose(1, 2)
        value = value.reshape(batch*self.columns, self.rows, dimension)
        attended = self.column(value, value, value, need_weights=False)[0]
        cells = cells+attended.reshape(batch, self.columns, self.rows, dimension).transpose(1, 2).reshape_as(cells)
        cells = cells+self.context(self.norms[2](cells), context, context, key_padding_mask=~context_valid, need_weights=False)[0]
        return cells+self.ff(self.norms[3](cells))


class TetrisStateNet(nn.Module):
    def __init__(self, config=ModelConfig()):
        super().__init__()
        self.config = config
        dimension = config.dimension
        self.encoder = ImageEncoder(config)
        self.board_query = nn.Parameter(torch.randn(1, 1, dimension)*.02)
        self.board_reader = nn.MultiheadAttention(dimension, config.heads, batch_first=True)
        self.initial_box = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, 4))
        self.box_refine = nn.ModuleList(nn.Linear(dimension, 4) for _ in range(config.refinements))
        self.row_embedding = nn.Embedding(config.visible_rows+config.spawn_rows, dimension)
        self.column_embedding = nn.Embedding(config.columns, dimension)
        row, column = torch.meshgrid(torch.arange(config.visible_rows+config.spawn_rows), torch.arange(config.columns), indexing="ij")
        self.register_buffer("coordinates", torch.stack((column+.5, row+.5), -1).reshape(-1, 2))
        self.register_buffer("row_ids", row.flatten())
        self.register_buffer("column_ids", column.flatten())
        y, x = torch.meshgrid(torch.tensor([-.35, 0., .35]), torch.tensor([-.35, 0., .35]), indexing="ij")
        self.register_buffer("base_offsets", torch.stack((x, y), -1).reshape(1, 1, 9, 2))
        self.offsets = nn.Linear(dimension, 18)
        nn.init.zeros_(self.offsets.weight)
        nn.init.zeros_(self.offsets.bias)
        self.sample_reader = nn.Sequential(nn.Linear(9*dimension, dimension), nn.LayerNorm(dimension))
        self.decoders = nn.ModuleList(AxialDecoder(config) for _ in range(config.blocks))
        self.occupancy = nn.Linear(dimension, 1)
        self.semantic = nn.Linear(dimension, 8)
        self.observability = nn.Linear(dimension, 1)
        self.registration = nn.Linear(dimension, 1)
        self.layer_queries = nn.Parameter(torch.randn(1, 2, dimension)*.02)
        self.layer_reader = nn.MultiheadAttention(dimension, config.heads, batch_first=True)
        self.layer_type = nn.Linear(dimension, 8)  # absent plus seven types
        self.layer_known = nn.Linear(dimension, 1)
        self.pose_rotation = nn.Linear(dimension, 4)
        self.pose_column = nn.Linear(dimension, config.columns)
        self.pose_row = nn.Linear(dimension, config.visible_rows+config.spawn_rows)
        self.components = None
        if config.component_capacity:
            from .components import ComponentDecoder
            self.components = ComponentDecoder(dimension, config.heads)
        # Last initialization preserves common seeded tensors in paired studies.
        self.layer_evidence = None
        if config.layer_evidence:
            from .layer_evidence import LayerEvidence
            self.layer_evidence = LayerEvidence(dimension)

    def forward(self, pixels, valid):
        check(pixels.ndim == 4 and pixels.shape[1] == 3 and pixels.is_floating_point(), "expected BCHW floating RGB")
        batch, _, height, width = pixels.shape
        check(1 <= batch <= self.config.max_batch and height*width <= self.config.max_pixels,
              "input batch/pixel budget exceeded")
        check(min(height, width) >= 16 and height % 16 == 0 and width % 16 == 0, "pad image dimensions to multiples of 16")
        check(valid.shape == (batch, 1, height, width) and valid.dtype == torch.bool, "valid-pixel mask required")
        check(bool(valid.flatten(1).any(1).all()) and bool(torch.isfinite(pixels).all()), "nonfinite/empty input")
        detail, coarse, context_valid = self.encoder(pixels, valid)
        context = coarse.flatten(2).transpose(1, 2)
        context = context+spatial_encoding(*coarse.shape[-2:], self.config.dimension, pixels.device, pixels.dtype)
        context_valid = context_valid.flatten(1)
        board = self.board_reader(self.board_query.expand(batch, -1, -1), context, context,
                                  key_padding_mask=~context_valid, need_weights=False)[0]
        raw_box = self.initial_box(board[:, 0])
        cells = (self.row_embedding(self.row_ids)+self.column_embedding(self.column_ids))[None].expand(batch, -1, -1)
        stages = []
        for index, decoder in enumerate(self.decoders):
            if index < len(self.box_refine):
                raw_box = raw_box+.2*torch.tanh(self.box_refine[index](board[:, 0]+cells.mean(1)))
            lattice = decode_lattice(raw_box)
            offsets = self.base_offsets+.1*torch.tanh(self.offsets(cells+board)).reshape(batch, -1, 9, 2)
            sampled, points = sample_features(detail, lattice, self.coordinates, offsets)
            cells = decoder(cells+self.sample_reader(sampled.flatten(2)), context, context_valid)
            combined = torch.cat((cells, context), dim=1)
            combined_valid = torch.cat((torch.ones(cells.shape[:2], dtype=torch.bool, device=pixels.device), context_valid), dim=1)
            layers = self.layer_reader(self.layer_queries.expand(batch, -1, -1)+board,
                                       combined, combined, key_padding_mask=~combined_valid, need_weights=False)[0]
            layer_cells = None
            if self.layer_evidence is not None:
                layer_cells, local_layers = self.layer_evidence(cells)
                layers = layers+local_layers
            rows, columns = self.config.visible_rows+self.config.spawn_rows, self.config.columns
            stages.append({"lattice": lattice, "occupancy": self.occupancy(cells).reshape(batch, rows, columns),
                           "semantic": self.semantic(cells).reshape(batch, rows, columns, 8),
                           "observability": self.observability(cells).reshape(batch, rows, columns),
                           "registration": self.registration(board[:, 0]).squeeze(-1),
                           "layer_type": self.layer_type(layers), "layer_known": self.layer_known(layers).squeeze(-1),
                           "pose_rotation": self.pose_rotation(layers), "pose_column": self.pose_column(layers),
                           "pose_row": self.pose_row(layers), "sample_points": points, "sample_offsets": offsets})
            if layer_cells is not None:
                stages[-1]["layer_cells"] = layer_cells
            if self.components is not None:
                stages[-1]["components"] = self.components(combined, combined_valid, board, capacity=self.config.component_capacity)
        return {"stages": stages, "final": stages[-1]}
