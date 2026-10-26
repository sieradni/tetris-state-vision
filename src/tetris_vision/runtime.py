"""M2 checkpoint inspection. Every result requires review; this is not release parsing."""
from pathlib import Path

import torch
from torch.nn import functional as F

from .checkpoint import load_checkpoint
from .losses import PIECE_NAMES, TYPE_NAMES
from .preprocess import prepare_image
from .shapes import rotated_shape
from .state import Board, Cell, FallingPiece, Icon, Lattice, Queue, VisionState


def _coverage(prepared, points):
    grid = points.reshape(1, -1, 1, 2)
    sampled = F.grid_sample(prepared.valid[None].float(), 2*grid-1, align_corners=False)
    return sampled.reshape(points.shape[:-1])


@torch.no_grad()
def decode_partial(output, prepared, *, model_id):
    """Deterministic serialization only; no game-playability repairs or acceptance."""
    geometry = output["lattice"][0].detach().cpu()
    columns, rows = 10, 20
    corners = torch.tensor([[0., 0.], [10., 0.], [0., 20.], [10., 20.]])
    projected = geometry[0]+corners[:, :1]*geometry[1]+corners[:, 1:]*geometry[2]
    limits = torch.tensor([prepared.resized_size[0]/prepared.padded_size[0],
                           prepared.resized_size[1]/prepared.padded_size[1]])
    inner = torch.tensor([[.15, .15], [9.85, .15], [.15, 19.85], [9.85, 19.85]])
    interior = geometry[0]+inner[:, :1]*geometry[1]+inner[:, 1:]*geometry[2]
    registered = (bool(output["registration"][0] > 0) and bool((projected >= 0).all())
                  and bool((projected <= limits).all()) and bool((_coverage(prepared, interior) >= .99).all()))
    factors = torch.tensor([prepared.padded_size[0]/prepared.scale_x, prepared.padded_size[1]/prepared.scale_y])
    source_geometry = geometry*factors
    lattice = Lattice(*(tuple(vector.tolist()) for vector in source_geometry)) if registered else None
    row, column = torch.meshgrid(torch.arange(26), torch.arange(10), indexing="ij")
    centers = torch.stack((column+.5, row+.5), -1)
    yy, xx = torch.meshgrid(torch.tensor([-.35, 0., .35]), torch.tensor([-.35, 0., .35]), indexing="ij")
    offsets = torch.stack((xx, yy), -1).reshape(1, 1, 9, 2)
    locations = centers[:, :, None]+offsets
    points = geometry[0]+locations[..., :1]*geometry[1]+locations[..., 1:]*geometry[2]
    cell_visible = (_coverage(prepared, points).min(-1).values >= .99)
    occupancy = output["occupancy"][0].detach().cpu()
    visibility = output["observability"][0].detach().cpu()
    probabilities = output["semantic"][0].detach().cpu().softmax(-1)
    board_rows = []
    for r in range(rows):
        line = []
        for c in range(columns):
            if not registered or not bool(cell_visible[r, c]) or not bool(visibility[r, c] > 0):
                line.append(Cell(None, visibility="hidden"))
                continue
            occupied = bool(occupancy[r, c] > 0)
            certainty, type_index = probabilities[r, c].max(-1)
            semantic = TYPE_NAMES[int(type_index)] if occupied and float(certainty) >= .5 else None
            line.append(Cell(occupied, semantic))
        board_rows.append(tuple(line))

    def layer(index):
        if not registered or not bool(output["layer_known"][0, index] > 0):
            return FallingPiece("unknown")
        identity = int(output["layer_type"][0, index].argmax())
        if identity == 0:
            if index == 0 and not bool(cell_visible.all()):
                return FallingPiece("unknown")
            return FallingPiece("absent")
        piece = PIECE_NAMES[identity-1]
        rotations = output["pose_rotation"][0, index].detach().cpu().log_softmax(-1)
        column_scores = output["pose_column"][0, index].detach().cpu().log_softmax(-1)
        row_scores = output["pose_row"][0, index].detach().cpu().log_softmax(-1)
        candidates = []
        for rotation in range(4):
            shape = rotated_shape(piece, rotation)
            for r in range(26-max(rr for _, rr in shape)):
                for c in range(10-max(cc for cc, _ in shape)):
                    score = float(rotations[rotation]+row_scores[r]+column_scores[c])
                    candidates.append((score, rotation, c, r, shape))
        _, _, c, r, shape = max(candidates, key=lambda item: item[0])
        cells = tuple((c+cc, r+rr) for cc, rr in shape)
        if not all(bool(cell_visible[rr, cc]) for cc, rr in cells):
            return FallingPiece("unknown")
        return FallingPiece("observed", piece, cells)

    return VisionState(Board(tuple(board_rows), lattice), layer(0), layer(1),
                       Queue((), "unknown", False, False), Icon("unknown"), "needs_review",
                       producer="TetrisStateNet-M2-uncalibrated", model_id=model_id)


class ImageParser:
    """One-image API around an owned M2 artifact; never returns accepted."""
    def __init__(self, model, record):
        self.model = model.cpu().eval()
        self.record = record
        self.max_side = record.get("metadata", {}).get("arguments", {}).get("max_side", 960)

    @classmethod
    def from_checkpoint(cls, path: str | Path):
        return cls(*load_checkpoint(path))

    @torch.no_grad()
    def parse_image(self, image):
        prepared = prepare_image(image, max_side=self.max_side)
        if not bool(prepared.valid.any()):
            return VisionState(None, FallingPiece("unknown"), FallingPiece("unknown"),
                               Queue((), "unknown", False, False), Icon("unknown"), "unsupported",
                               producer="TetrisStateNet-M2-uncalibrated", model_id=self.record["weights_sha256"])
        output = self.model(prepared.pixels[None], prepared.valid[None])["final"]
        return decode_partial(output, prepared, model_id=self.record["weights_sha256"])
