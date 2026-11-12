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
    component_identity_readout: str = "separate"
    component_endpoint_term: str = "joint"
    layer_evidence: bool = False
    occupancy_context: int = 0
    occupancy_detail: int = 0
    occupancy_detail_stencil: int = 0
    occupancy_resolution: int = 0
    occupancy_deform: int = 0
    lattice_support: int = 0
    lattice_residual: int = 0
    lattice_board_refine: int = 0
    lattice_board_refine: int = 0
    occupancy_lattice_free: int = 0
    occupancy_column: int = 0
    occupancy_offset_rows: int = 0

    def __post_init__(self):
        check(type(self.heads) is int and 1 <= self.heads <= 8, "invalid attention head count")
        check(len(self.widths) == 4 and all(type(c) is int and 8 <= c <= 256 and c % 8 == 0 for c in self.widths), "invalid encoder widths")
        check(16 <= self.dimension <= 256 and self.dimension % 4 == 0 and self.dimension % self.heads == 0, "invalid attention dimension")
        check(1 <= self.blocks <= 6 and 0 <= self.refinements <= min(3, self.blocks), "bounded decoder configuration required")
        check(self.visible_rows == 20 and self.spawn_rows == 6 and self.columns == 10, "prototype capacity is one 10x20 board plus six spawn rows")
        check(1 <= self.max_batch <= 4 and 4096 <= self.max_pixels <= 1536**2, "invalid work capacity")
        check(type(self.component_capacity) is int and 0 <= self.component_capacity <= 64,
              "component capacity must be0 (M2) or1..64 (M3)")
        check(self.component_identity_readout in ("separate", "shared", "prototype", "positional") and
              (self.component_capacity or self.component_identity_readout == "separate"),
              "identity readout must be separate, shared with the cell-type head, per-piece image queries, "
              "or per-position image queries")
        from .components import ENDPOINT_TERMS
        check(self.component_endpoint_term in ENDPOINT_TERMS and
              (self.component_endpoint_term == "joint" or
               (self.component_capacity and self.component_identity_readout in ("prototype", "positional"))) and
              (self.component_endpoint_term != "length" or self.component_identity_readout == "positional"),
              "separate or length endpoint term requires the joint heads and a declared identity read-out, "
              "and length requires the positional read-out")
        check(type(self.layer_evidence) is bool, "layer evidence must be an explicit boolean")
        check(type(self.occupancy_context) is int and self.occupancy_context in (0, 1),
              "occupancy context is0 (published read-out) or1 (declared neighborhood read-out)")
        check(type(self.occupancy_detail) is int and self.occupancy_detail in (0, 1),
              "occupancy detail is0 (published read-out) or1 (declared high-resolution detail tap)")
        check(type(self.occupancy_detail_stencil) is int and self.occupancy_detail_stencil in (0, 1),
              "occupancy detail stencil is0 (single center sample) or1 (declared 2x2 in-cell stencil)")
        check(not self.occupancy_detail_stencil or self.occupancy_detail == 1,
              "the detail stencil is a declared form of the detail tap, not a separate read-out")
        check(type(self.occupancy_resolution) is int and self.occupancy_resolution in (0, 1),
              "occupancy resolution is 0 (stride-2 sample) or 1 (declared stride-1 sample)")
        check(not self.occupancy_resolution or self.occupancy_detail == 1,
              "the resolution lever is a declared form of the detail tap, not a separate read-out")
        check(type(self.occupancy_deform) is int and self.occupancy_deform in (0, 1, 2, 3),
              "occupancy deform is 0 (the tap samples at the decoded cell), 1 (one-cell field), "
              "2 (coarse-to-fine field reaching four cells) or 3 (the same field de-meaned across "
              "the board, so it cannot express a translation)")
        check(not self.occupancy_deform or self.occupancy_detail == 1,
              "the registration lever moves the detail tap's sample, so it needs that tap")
        check(type(self.lattice_support) is int and self.lattice_support in (0, 1),
              "lattice support is 0 (published box head) or 1 (declared support-moment residual)")
        check(type(self.lattice_residual) is int and self.lattice_residual in (0, 1),
              "lattice residual is 0 (published shared decode) or 1 (declared per-stage residual)")
        check(type(self.lattice_board_refine) is int and self.lattice_board_refine in (0, 1),
              "lattice board refinement is 0 (published shared decode) or 1 (declared per-board capacity)")
        check(type(self.lattice_board_refine) is int and self.lattice_board_refine in (0, 1),
              "lattice board refinement is 0 (published shared decode) or 1 (declared per-board capacity)")
        check(type(self.occupancy_lattice_free) is int and self.occupancy_lattice_free in (0, 1),
              "occupancy lattice-free readout is 0 (published read-out) or 1 (declared gated channel)")
        check(type(self.occupancy_column) is int and self.occupancy_column in (0, 1),
              "occupancy column read-out is 0 (published read-out) or 1 (declared per-column "
              "stack context)")
        check(type(self.occupancy_offset_rows) is int and
              -2 <= self.occupancy_offset_rows <= 2,
              "occupancy offset rows is a declared constant in [-2, 2] cell rows, 0 being the "
              "published sampling point")
        check(not self.occupancy_offset_rows or self.occupancy_detail == 1,
              "the sampling offset moves the detail tap's sample, so it needs that tap")

    @classmethod
    def pilot(cls):
        return cls(widths=(16, 32, 64, 96), dimension=64, heads=4, blocks=2)


# Declared trunk shapes. The published and pilot shapes keep their historical identity;
# any other declared shape is a second architecture identity with its own name.
PUBLISHED_WIDTHS = (48, 96, 192, 256)
PUBLISHED_DIMENSION = 192
PILOT_WIDTHS = (16, 32, 64, 96)
PILOT_DIMENSION = 64


def column_context(cells, rows, columns):
    """Per-cell view of its own column: mean feature below and above, and the spans they cover.

    Rows are bottom-first everywhere in this project, so "below" a cell is the strictly smaller row
    index - the material an occupancy label at that cell is really reporting. Built from cumulative
    sums, so it costs one [batch, cells, 2*dimension+2] intermediate and adds no learnable tensor of
    its own; the projection that reads it is zero-initialized, which keeps step zero exactly the
    published function.
    """
    batch, count, dimension = cells.shape
    if count != rows*columns:
        raise ValueError("column context requires exactly one feature per board cell")
    grid = cells.reshape(batch, rows, columns, dimension)
    cumulative = grid.cumsum(1)
    below = cumulative-grid
    above = grid.sum(1)[:, None]-cumulative
    index = torch.arange(rows, device=cells.device, dtype=cells.dtype)
    below_count = index.clamp_min(1).reshape(1, rows, 1, 1)
    above_count = (rows-index-1).clamp_min(1).reshape(1, rows, 1, 1)
    position = (index/(rows-1)).reshape(1, rows, 1, 1).expand(batch, rows, columns, 1)
    context = torch.cat((below/below_count, above/above_count, position, 1-position), -1)
    if context.shape[-1] != 2*dimension+2:
        raise ValueError("column context has an unexpected width")
    return context.reshape(batch, count, 2*dimension+2)


def trunk_suffix(config):
    """Empty for the published/pilot shapes, else the declared shape in the name."""
    shape = (tuple(config.widths), config.dimension)
    if shape in ((PUBLISHED_WIDTHS, PUBLISHED_DIMENSION), (PILOT_WIDTHS, PILOT_DIMENSION)):
        return ""
    return "-W"+"x".join(str(channel) for channel in config.widths)+"-D"+str(config.dimension)


def architecture_name(config):
    """Derived identity: capacity decides M2/M3, the declared trunk shape decides the suffix.

    Never a handwritten label: every reader rebuilds this from the configuration it
    verified, so a widened configuration cannot be reported under the published name.
    """
    return ("TetrisStateNet-M3" if config.component_capacity else "TetrisStateNet-M2")+trunk_suffix(config)


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
        return detail, context, weight[:, 0] > 0, first


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


LATTICE_RESIDUAL_SCALE = .25
# Declared registration lever: a per-cell field may move its sample by at most one cell in each
# axis. That is larger than any sub-cell correction and still local, so the field can follow
# misalignment the global box would have to move for, without letting a cell sample a different cell.
DEFORM_LIMIT = 1.0
# Declared widened reach: the measured placement residual is 2.57 cells on average, which the one-cell
# field cannot express by construction. A coarse level predicted from a pooled map - eight times the
# feature stride, so its 3x3 window sees roughly eight cells instead of one - may move the sample by
# up to three cells, and the fine level adds one more, so the pair covers the measured residual while
# neither level alone can wander. `first` is stride 2, so a 3x3 convolution at that resolution sees
# about a tenth of a cell: the wider reach has to come with a coarser context, not just a bigger bound.
DEFORM_WIDE_LIMIT = 3.0
DEFORM_REACH = 4.0
# Declared de-meaned reach: subtracting the board's mean offset can move any single cell by as much
# as the sum of the level bounds (one shifted endpoint plus the mean), so the structural bound of the
# de-meaned arm is twice the free arm's. It is not a clamp: clamping after de-meaning would put the
# mean back, so the arm's bound is declared to cover the de-meaned sum instead.
DEFORM_DEMEAN_REACH = 2*DEFORM_REACH
DEFORM_LIMITS = (DEFORM_LIMIT, DEFORM_WIDE_LIMIT)


def deform_offsets(samples, limit=DEFORM_LIMIT):
    """Bounded per-cell sample offset in cell units; exactly zero for a zero field."""
    if limit not in DEFORM_LIMITS:
        raise ValueError("only the declared deformation limits are available")
    return torch.tanh(samples)*limit


def apply_lattice_residual(lattice, residual):
    """Declared per-stage lattice residual in cell units and log-scale; the identity at zero."""
    check(residual.shape == (lattice.shape[0], 4),
          "a lattice residual is one four-vector per board")
    value = LATTICE_RESIDUAL_SCALE*torch.tanh(residual)
    cells, scale = value[:, :2], torch.exp(value[:, 2:])
    origin = lattice[:, 0]+cells[:, :1]*lattice[:, 1]+cells[:, 1:]*lattice[:, 2]
    return torch.stack((origin, lattice[:, 1]*scale[:, :1],
                        lattice[:, 2]*scale[:, 1:]), dim=1)


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
        # Declared translation lever: the box head sees one global token, while the
        # measured placement residual is a board-dependent origin shift. This
        # zero-initialized projection adds the coarse feature map's spatial moments
        # (centroid and spread), so step zero is exactly the published function and the
        # paired control starts from the same tensors.
        self.support_box = None
        if config.lattice_support:
            self.support_box = nn.Linear(5, 4)
            nn.init.zeros_(self.support_box.weight)
            nn.init.zeros_(self.support_box.bias)
        # Declared per-stage lattice residual: the shared affine decode ties the origin to the
        # extent through one sigmoid and is refined before that sigmoid for only `refinements`
        # stages. This zero-initialized projection acts in decoded cell space at every stage, so
        # a stage's boundary can be pinned without moving the shared box. Zero init keeps step
        # zero exactly the published function.
        self.lattice_residual = None
        if config.lattice_residual:
            self.lattice_residual = nn.ModuleList(nn.Linear(4, 4) for _ in range(config.blocks))
            for layer in self.lattice_residual:
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
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
            if config.component_identity_readout == "shared":
                # Declared lever: piece identity reads through the cell-type head.
                self.components.share_type_readout(self.semantic)
            elif config.component_identity_readout == "prototype":
                # Declared lever: per-class image queries, so discrimination is possible
                # by construction rather than through one collapsed role vector.
                self.components.use_prototype_readout()
            elif config.component_identity_readout == "positional":
                # Declared lever: per-position image queries on top of the validated class
                # and hold banks, so the position axis can carry information too.
                self.components.use_positional_queries()
            if config.component_endpoint_term == "separate":
                # Declared lever: the endpoint is its own binary decision, outside the
                # piece softmax, so a constant piece class cannot hide behind it.
                self.components.use_factorized_endpoint()
            elif config.component_endpoint_term == "length":
                # Declared lever: one queue-length decision per case replaces the
                # per-position endpoint question.
                self.components.use_length_distribution()
        # Last initialization preserves common seeded tensors in paired studies.
        self.layer_evidence = None
        if config.layer_evidence:
            from .layer_evidence import LayerEvidence
            self.layer_evidence = LayerEvidence(dimension)
        # Declared representation lever: the occupancy read-out sees its eight lattice
        # neighbors, so a cell's decision can use the local edge it sits on. The final
        # projection is zero-initialized, which makes step zero exactly the published
        # function and gives the paired control the same starting tensors.
        self.occupancy_detail_tap = None
        # Declared stencil form: the four quarter-cell positions of a 2x2 grid inside
        # each cell, in the lattice's own units. Kept out of the state dict so both
        # arms carry identical keys except the projection itself.
        self.register_buffer("detail_stencil", torch.tensor([[-.25, -.25], [.25, -.25],
                                                             [-.25, .25], [.25, .25]]).reshape(1, 1, 4, 2),
                             persistent=False)
        if config.occupancy_detail:
            # Declared precision lever: the occupancy read-out also sees the stride-2
            # first-stage feature at the cell center (or, in the declared stencil form,
            # its four quarter-cell samples), a sharper sample than the decoder latent.
            # The projection is zero-initialized, so step zero is exactly the published
            # function and the paired control starts from the same tensors.
            samples = 4 if config.occupancy_detail_stencil else 1
            self.occupancy_detail_tap = nn.Linear(config.widths[0]*samples, dimension)
            nn.init.zeros_(self.occupancy_detail_tap.weight)
            nn.init.zeros_(self.occupancy_detail_tap.bias)
        # Declared sampling geometry: a constant shift of every tap sample, in cell units, rows
        # bottom-first. Unlike the deformation field this is not learned - it is a declaration about
        # where the read looks - so it is a non-persistent buffer and the state dict is unchanged;
        # at 0 the added constant is exactly zero and the arm is the published function.
        self.register_buffer("tap_offset",
                             torch.tensor([[[[0.0, float(config.occupancy_offset_rows)]]]]),
                             persistent=False)
        self.occupancy_resolution_branch = None
        if config.occupancy_resolution:
            # Declared resolution lever: a stride-1 source for the same zero-initialized
            # projection, so the tap can read pixel resolution instead of the encoder's
            # stride-2 feature. It exists only on the levered arm and the projection stays
            # zero-initialized, which keeps step zero exactly the published function.
            self.occupancy_resolution_branch = nn.Conv2d(3, config.widths[0], 3, padding=1)
        self.occupancy_neighborhood = None
        if config.occupancy_context:
            self.occupancy_neighborhood = nn.Sequential(
                nn.Conv2d(dimension, dimension, 3, padding=1, bias=False),
                nn.GroupNorm(8, dimension), nn.SiLU(),
                nn.Conv2d(dimension, dimension, 1))
            nn.init.zeros_(self.occupancy_neighborhood[-1].weight)
            nn.init.zeros_(self.occupancy_neighborhood[-1].bias)
        # Declared read-out lever: a lattice-agnostic occupancy channel. It reads the encoder's
        # image-level context and fixed cell coordinates, never the predicted lattice, so it is
        # identical at every decoder stage. The gate starts at exactly zero and the projections are
        # zero-initialized, so step zero is bitwise the published read-out; as the gate moves to 1
        # the published lattice-sampled path (and its detail tap) is replaced by this channel.
        # Constructed last so the published tensors draw the same seeded values in paired studies.
        self.occupancy_lattice_free = None
        self.occupancy_lattice_gate = None
        if config.occupancy_lattice_free:
            self.occupancy_lattice_free = nn.Sequential(nn.Linear(dimension, dimension), nn.SiLU(),
                                                        nn.Linear(dimension, 1))
            for layer in (self.occupancy_lattice_free[0], self.occupancy_lattice_free[2]):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
            self.occupancy_lattice_gate = nn.Parameter(torch.zeros(1))
        # Declared capacity lever: one correction of the decoded lattice per board, predicted from
        # the board token and applied identically at every stage - origin offset in cells along the
        # decoded axes plus a log-scale per axis. It is board-conditioned rather than per-stage, so
        # the stages share one correction. Every tensor starts at exactly zero, which keeps step zero
        # bitwise the published function; constructed last so published tensors keep their seeds.
        self.lattice_board_refine = None
        if config.lattice_board_refine:
            self.lattice_board_refine = nn.Sequential(nn.Linear(dimension, 64), nn.SiLU(),
                                                      nn.Linear(64, 4))
            for layer in (self.lattice_board_refine[0], self.lattice_board_refine[2]):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
        # Declared capacity lever: one correction of the decoded lattice per board, predicted from
        # the board token and applied identically at every stage - origin offset in cells along the
        # decoded axes plus a log-scale per axis. It is board-conditioned rather than per-stage, so
        # the stages share one correction. Every tensor starts at exactly zero, which keeps step zero
        # bitwise the published function; constructed last so published tensors keep their seeds.
        self.lattice_board_refine = None
        if config.lattice_board_refine:
            self.lattice_board_refine = nn.Sequential(nn.Linear(dimension, 64), nn.SiLU(),
                                                      nn.Linear(64, 4))
            for layer in (self.lattice_board_refine[0], self.lattice_board_refine[2]):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
        # Declared registration lever: a per-cell deformation field, predicted from the same
        # first-stage feature the tap reads and sampled at the decoded cell centres, moves each
        # tap sample by at most one cell. The field's own convolution is zero-initialized, so step
        # zero samples exactly where the published function samples and the paired control starts
        # from identical tensors; unlike a box or lattice correction it is local, so it can follow
        # per-cell misalignment without moving the placement the geometry term is judged on.
        self.occupancy_deform = None
        self.occupancy_deform_coarse = None
        if config.occupancy_deform:
            self.occupancy_deform = nn.Conv2d(config.widths[0], 2, 3, padding=1)
            nn.init.zeros_(self.occupancy_deform.weight)
            nn.init.zeros_(self.occupancy_deform.bias)
            if config.occupancy_deform in (2, 3):
                # Coarse level: predicted at the pooled resolution so its window covers a board
                # neighbourhood, then upsampled to the fine map and sampled at the same cell
                # centres. Zero-initialized like the fine level, so step zero is the published
                # function and the fine arm is a strict prefix of this one.
                self.occupancy_deform_coarse = nn.Conv2d(config.widths[0], 2, 3, padding=1)
                nn.init.zeros_(self.occupancy_deform_coarse.weight)
                nn.init.zeros_(self.occupancy_deform_coarse.bias)
        # Declared read-out lever: each cell's occupancy feature also sees the column it stands in,
        # as the mean cell feature strictly below and strictly above it plus the spans they cover.
        # The evidence that motivates it (M4_OCCUPANCY_PLACED.md) is that the objective's only
        # profitable displacement was a read of the material below the cell; putting that material
        # beside the cell means no displacement has to be bought at all. Zero-initialized, so step
        # zero is exactly the published function and the paired control starts from the same
        # tensors.
        self.occupancy_column = None
        if config.occupancy_column:
            self.occupancy_column = nn.Linear(2*dimension+2, dimension)
            nn.init.zeros_(self.occupancy_column.weight)
            nn.init.zeros_(self.occupancy_column.bias)

    @staticmethod
    def support_moments(coarse):
        """Normalized spatial moments of the coarse feature energy, shape [batch, 5]."""
        energy = coarse.mean(1).relu()
        total = energy.sum((1, 2), keepdim=True).clamp_min(1e-6)
        weights = energy/total
        height, width = energy.shape[-2:]
        xs = ((torch.arange(width, device=energy.device, dtype=energy.dtype)+.5)/width*2-1).view(1, 1, width)
        ys = ((torch.arange(height, device=energy.device, dtype=energy.dtype)+.5)/height*2-1).view(1, height, 1)
        column = (weights*xs).sum((1, 2))
        row = (weights*ys).sum((1, 2))
        spread_x = (weights*(xs-column[:, None, None])**2).sum((1, 2)).clamp_min(0).sqrt()
        spread_y = (weights*(ys-row[:, None, None])**2).sum((1, 2)).clamp_min(0).sqrt()
        return torch.stack((column, row, spread_x, spread_y,
                            torch.log1p(energy.mean((1, 2)))), -1)

    def forward(self, pixels, valid):
        check(pixels.ndim == 4 and pixels.shape[1] == 3 and pixels.is_floating_point(), "expected BCHW floating RGB")
        batch, _, height, width = pixels.shape
        check(1 <= batch <= self.config.max_batch and height*width <= self.config.max_pixels,
              "input batch/pixel budget exceeded")
        check(min(height, width) >= 16 and height % 16 == 0 and width % 16 == 0, "pad image dimensions to multiples of 16")
        check(valid.shape == (batch, 1, height, width) and valid.dtype == torch.bool, "valid-pixel mask required")
        check(bool(valid.flatten(1).any(1).all()) and bool(torch.isfinite(pixels).all()), "nonfinite/empty input")
        detail, coarse, context_valid, first = self.encoder(pixels, valid)
        context = coarse.flatten(2).transpose(1, 2)
        context = context+spatial_encoding(*coarse.shape[-2:], self.config.dimension, pixels.device, pixels.dtype)
        context_valid = context_valid.flatten(1)
        board = self.board_reader(self.board_query.expand(batch, -1, -1), context, context,
                                  key_padding_mask=~context_valid, need_weights=False)[0]
        raw_box = self.initial_box(board[:, 0])
        if self.support_box is not None:
            raw_box = raw_box+self.support_box(self.support_moments(coarse))
        refinement = (None if self.lattice_board_refine is None
                      else self.lattice_board_refine(board[:, 0]))
        refinement = (None if self.lattice_board_refine is None
                      else self.lattice_board_refine(board[:, 0]))
        cells = (self.row_embedding(self.row_ids)+self.column_embedding(self.column_ids))[None].expand(batch, -1, -1)
        stages = []
        for index, decoder in enumerate(self.decoders):
            if index < len(self.box_refine):
                raw_box = raw_box+.2*torch.tanh(self.box_refine[index](board[:, 0]+cells.mean(1)))
            lattice = decode_lattice(raw_box)
            if self.lattice_residual is not None:
                lattice = apply_lattice_residual(lattice, self.lattice_residual[index](raw_box))
            if refinement is not None:
                lattice = apply_lattice_residual(lattice, refinement)
            if refinement is not None:
                lattice = apply_lattice_residual(lattice, refinement)
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
            occupancy_features = cells
            occupancy_deform_offsets = None
            occupancy_applied_offsets = None
            if self.occupancy_detail_tap is not None:
                offsets = (self.detail_stencil if self.config.occupancy_detail_stencil else
                           torch.zeros(1, 1, 1, 2, device=pixels.device, dtype=pixels.dtype))
                if self.occupancy_deform is not None:
                    zero = torch.zeros(1, 1, 1, 2, device=pixels.device, dtype=pixels.dtype)
                    shift = deform_offsets(sample_features(
                        self.occupancy_deform(first), lattice, self.coordinates, zero
                    )[0].reshape(batch, rows*columns, 1, 2))
                    if self.occupancy_deform_coarse is not None:
                        pooled = F.avg_pool2d(first, 8, ceil_mode=True)
                        coarse = F.interpolate(self.occupancy_deform_coarse(pooled),
                                               size=first.shape[-2:], mode="bilinear",
                                               align_corners=False)
                        shift = shift+deform_offsets(sample_features(
                            coarse, lattice, self.coordinates, zero
                        )[0].reshape(batch, rows*columns, 1, 2), DEFORM_WIDE_LIMIT)
                        shift = shift.clamp(-DEFORM_REACH, DEFORM_REACH)
                    if self.config.occupancy_deform == 3:
                        # Declared constant-mode bar: the field is de-meaned across the board before it
                        # is applied, so the board-wide translation is exactly zero by construction and
                        # only per-cell displacement remains expressible. Nothing in the objective can
                        # buy a re-centred read-out through this arm, and step zero is still the
                        # published function because a zero field has zero mean.
                        shift = shift-shift.mean(dim=1, keepdim=True)
                    occupancy_deform_offsets = shift.reshape(batch, rows*columns, 2)
                    offsets = offsets+shift
                # The declared constant geometry, added after any learned field so the field keeps
                # meaning "what was learned" and this keeps meaning "what was declared".
                offsets = offsets+self.tap_offset
                centers = sample_features(first, lattice, self.coordinates, offsets)[0]
                sample = centers.flatten(2)
                # The tap's offsets as applied, so a measurement can bind "what the arm wrote"
                # to "what it sampled" instead of assuming them equal. Only a single-sample tap
                # applies one vector per cell; the stencil reads several fixed samples around the
                # cell, so a single applied vector does not exist and is not reported (V-118).
                if not self.config.occupancy_detail_stencil:
                    occupancy_applied_offsets = (offsets+torch.zeros(
                        batch, rows*columns, 1, 2, device=pixels.device, dtype=pixels.dtype
                    )).reshape(batch, rows*columns, 2)
                if self.occupancy_resolution_branch is not None:
                    fine = self.occupancy_resolution_branch((2*pixels-1)*valid)
                    fine = sample_features(fine, lattice, self.coordinates, torch.zeros(
                        1, 1, 1, 2, device=pixels.device, dtype=pixels.dtype))[0]
                    sample = sample+fine.flatten(2)
                occupancy_features = occupancy_features+self.occupancy_detail_tap(sample)
            if self.occupancy_column is not None:
                # Declared read-out: the cell's own column, summarised below and above it. It reads
                # the decoder's cell features at the cell, so it never moves a sample.
                occupancy_features = occupancy_features+self.occupancy_column(
                    column_context(cells, rows, columns))
            if self.occupancy_neighborhood is not None:
                neighborhood = cells.reshape(batch, rows, columns, self.config.dimension).permute(0, 3, 1, 2)
                # Declared stacking lever: the two residuals are summed, so turning both taps on
                # adds their information instead of letting the context block overwrite the detail
                # tap. With either tap alone this is byte-identical to the published composition.
                occupancy_features = occupancy_features+self.occupancy_neighborhood(neighborhood).permute(0, 2, 3, 1).reshape(batch, rows*columns, self.config.dimension)
            occupancy_logits = self.occupancy(occupancy_features).reshape(batch, rows, columns)
            if self.occupancy_lattice_free is not None:
                weight = context_valid.float()[:, :, None]
                pooled = (context*weight).sum(1)/weight.sum(1).clamp_min(1)
                free = ((self.row_embedding(self.row_ids)+self.column_embedding(self.column_ids))[None]
                        +pooled[:, None])
                gate = torch.tanh(self.occupancy_lattice_gate)
                occupancy_logits = ((1-gate)*occupancy_logits+
                                    gate*self.occupancy_lattice_free(free).reshape(batch, rows, columns))
            stages.append({"lattice": lattice, "occupancy": occupancy_logits,
                           "semantic": self.semantic(cells).reshape(batch, rows, columns, 8),
                           "observability": self.observability(cells).reshape(batch, rows, columns),
                           "registration": self.registration(board[:, 0]).squeeze(-1),
                           "layer_type": self.layer_type(layers), "layer_known": self.layer_known(layers).squeeze(-1),
                           "pose_rotation": self.pose_rotation(layers), "pose_column": self.pose_column(layers),
                           "pose_row": self.pose_row(layers), "sample_points": points, "sample_offsets": offsets})
            if occupancy_deform_offsets is not None:
                stages[-1]["occupancy_deform_offsets"] = occupancy_deform_offsets
            if occupancy_applied_offsets is not None:
                stages[-1]["occupancy_applied_offsets"] = occupancy_applied_offsets
            if layer_cells is not None:
                stages[-1]["layer_cells"] = layer_cells
            if self.components is not None:
                stages[-1]["components"] = self.components(combined, combined_valid, board, capacity=self.config.component_capacity)
        if refinement is not None:
            stages[-1]["board_refinement"] = refinement
        if refinement is not None:
            stages[-1]["board_refinement"] = refinement
        return {"stages": stages, "final": stages[-1]}
