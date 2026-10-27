"""Image-supported M2 targets. Unknown values never become negative labels."""
import torch
from torch.nn import functional as F

from .shapes import normalize, rotated_shape
from .state import check

TYPE_NAMES = tuple("IOTSZJLG")
PIECE_NAMES = tuple("IOTSZJL")


def targets_for(state, prepared):
    check(state.board is None or (state.board.width == 10 and state.board.height == 20),
          "M2 target capacity is one 10x20 board")
    width, height = prepared.padded_size
    occupied = torch.zeros((20, 10))
    known = torch.zeros((20, 10), dtype=torch.bool)
    semantic = torch.zeros((20, 10), dtype=torch.long)
    typed = torch.zeros((20, 10), dtype=torch.bool)
    lattice = torch.tensor([[0., 1.], [.1, 0.], [0., -.05]])
    registered = state.board is not None and state.board.lattice is not None and bool(prepared.valid.any())
    readable_scale = False
    if registered:
        geometry = state.board.lattice.transformed(prepared.scale_x, prepared.scale_y)
        lattice = torch.tensor([geometry.origin, geometry.u, geometry.v])/torch.tensor([width, height])
        readable_scale = min(sum(x*x for x in geometry.u)**.5, sum(x*x for x in geometry.v)**.5) >= 10
        cell_valid = torch.ones((20, 10), dtype=torch.bool)
        # Account for source PNG transparency as well as letterbox padding. This
        # mask is based on target geometry; predicted geometry cannot hide loss.
        offsets = ((-.35, -.35), (0., -.35), (.35, -.35), (-.35, 0.), (0., 0.), (.35, 0.),
                   (-.35, .35), (0., .35), (.35, .35))
        grid = []
        for r in range(20):
            for c in range(10):
                points = []
                for dx, dy in offsets:
                    points.append([2*(geometry.origin[0]+(c+.5+dx)*geometry.u[0]+(r+.5+dy)*geometry.v[0])/width-1,
                                   2*(geometry.origin[1]+(c+.5+dx)*geometry.u[1]+(r+.5+dy)*geometry.v[1])/height-1])
                grid.append(points)
        coverage = F.grid_sample(prepared.valid[None].float(), torch.tensor([grid]), align_corners=False)
        cell_valid = (coverage[0, 0].min(-1).values >= .99).reshape(20, 10)
        for r, row in enumerate(state.board.rows):
            for c, cell in enumerate(row):
                if readable_scale and bool(cell_valid[r, c]) and cell.occupied is not None:
                    occupied[r, c], known[r, c] = float(cell.occupied), True
                    if cell.occupied and cell.semantic is not None:
                        semantic[r, c], typed[r, c] = TYPE_NAMES.index(cell.semantic), True
    layer_type = torch.zeros(2, dtype=torch.long)
    layer_known = torch.zeros(2, dtype=torch.bool)
    pose_known = torch.zeros(2, dtype=torch.bool)
    column = torch.zeros(2, dtype=torch.long)
    row = torch.zeros(2, dtype=torch.long)
    rotations = torch.zeros((2, 4), dtype=torch.bool)
    for index, layer in enumerate((state.active, state.ghost)):
        if not registered or not readable_scale or layer.status not in {"observed", "absent"}:
            continue
        layer_known[index] = True
        if layer.status == "absent" and not bool(prepared.valid[:, :prepared.resized_size[1], :prepared.resized_size[0]].all()):
            layer_known[index] = False
            continue
        if layer.status == "observed":
            left, bottom = min(c for c, _ in layer.cells), min(r for _, r in layer.cells)
            if not (0 <= left < 10 and 0 <= bottom < 26):
                layer_known[index] = False
                continue
            layer_grid = [[[2*(geometry.origin[0]+(c+.5+dx)*geometry.u[0]+(r+.5+dy)*geometry.v[0])/width-1,
                            2*(geometry.origin[1]+(c+.5+dx)*geometry.u[1]+(r+.5+dy)*geometry.v[1])/height-1]
                           for dx, dy in offsets] for c, r in layer.cells]
            coverage = F.grid_sample(prepared.valid[None].float(), torch.tensor([layer_grid]), align_corners=False)
            if not bool((coverage >= .99).all()):
                layer_known[index] = False
                continue
            layer_type[index] = 1+PIECE_NAMES.index(layer.piece)
            column[index], row[index], pose_known[index] = left, bottom, True
            rotations[index] = torch.tensor([normalize(layer.cells) == rotated_shape(layer.piece, rotation)
                                              for rotation in range(4)])
    return {key: value.unsqueeze(0) for key, value in {
        "lattice": lattice, "registered": torch.tensor(registered),
        "occupied": occupied, "known": known, "semantic": semantic, "typed": typed,
        "layer_type": layer_type, "layer_known": layer_known, "pose_known": pose_known,
        "column": column, "row": row, "rotations": rotations}.items()}


def masked_mean(value, mask):
    return (value*mask).sum()/mask.sum().clamp_min(1)


def balanced_occupancy(logits, target, mask):
    errors = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    positive, negative = mask & (target == 1), mask & (target == 0)
    positive_count, negative_count = positive.sum((1, 2)), negative.sum((1, 2))
    per_image = ((errors*positive).sum((1, 2))/positive_count.clamp_min(1)+
                 (errors*negative).sum((1, 2))/negative_count.clamp_min(1))
    per_image = per_image/((positive_count > 0).float()+(negative_count > 0).float()).clamp_min(1)
    return masked_mean(per_image, mask.any((1, 2)))


def geometry_error(predicted, target):
    # Four boundary anchors and all 200 centers, measured in source cell units.
    row, column = torch.meshgrid(torch.arange(20, device=predicted.device),
                                torch.arange(10, device=predicted.device), indexing="ij")
    centers = torch.stack((column+.5, row+.5), -1).reshape(-1, 2)
    anchors = torch.tensor([[0., 0.], [10., 0.], [0., 20.], [10., 20.]], device=predicted.device)
    coordinates = torch.cat((centers, anchors)).to(predicted.dtype)
    def points(lattice):
        return lattice[:, None, 0]+coordinates[None, :, :1]*lattice[:, None, 1]+coordinates[None, :, 1:]*lattice[:, None, 2]
    spacing = torch.stack((target[:, 1].norm(dim=-1), target[:, 2].norm(dim=-1)), -1).clamp_min(1e-6)
    return (points(predicted)-points(target))/spacing[:, None]


def stage_loss(output, target):
    error = geometry_error(output["lattice"], target["lattice"])
    geometry = masked_mean(F.smooth_l1_loss(error, torch.zeros_like(error), reduction="none").mean((1, 2)), target["registered"])
    occupancy = balanced_occupancy(output["occupancy"][:, :20], target["occupied"], target["known"])
    type_error = F.cross_entropy(output["semantic"][:, :20].permute(0, 3, 1, 2), target["semantic"], reduction="none")
    semantic = masked_mean(type_error, target["typed"])
    visibility = F.binary_cross_entropy_with_logits(output["observability"][:, :20], target["known"].float())
    registration = F.binary_cross_entropy_with_logits(output["registration"], target["registered"].float())
    layer_error = F.cross_entropy(output["layer_type"].transpose(1, 2), target["layer_type"], reduction="none")
    layers = masked_mean(layer_error, target["layer_known"])
    layer_visibility = F.binary_cross_entropy_with_logits(output["layer_known"], target["layer_known"].float())
    # Symmetric pieces have several equivalent rotations. Summing their
    # probability avoids teaching arbitrary labels and avoids inf*0 on unknowns.
    allowed = target["rotations"].clone()
    allowed[~allowed.any(-1)] = True
    rotation_error = -output["pose_rotation"].log_softmax(-1).masked_fill(~allowed, -torch.inf).logsumexp(-1)
    column_error = F.cross_entropy(output["pose_column"].transpose(1, 2), target["column"], reduction="none")
    row_error = F.cross_entropy(output["pose_row"].transpose(1, 2), target["row"], reduction="none")
    pose = masked_mean(rotation_error+column_error+row_error, target["pose_known"])
    return {"geometry": geometry, "occupancy": occupancy, "semantic": semantic,
            "visibility": visibility, "registration": registration, "layers": layers,
            "layer_visibility": layer_visibility, "pose": pose}


def joint_loss(outputs, target):
    per_stage = [stage_loss(stage, target) for stage in outputs["stages"]]
    terms = {key: sum(stage[key] for stage in per_stage)/len(per_stage) for key in per_stage[0]}
    # Pilot hypotheses, not tuned or established research hyperparameters.
    weights = {"geometry": 1., "occupancy": 1., "semantic": .25, "visibility": .25,
               "registration": .25, "layers": .5, "layer_visibility": .25, "pose": .25}
    total = sum(weights[key]*value for key, value in terms.items())
    return total, terms


@torch.no_grad()
def diagnostic_metrics(outputs, target):
    final = outputs["final"]
    known = target["known"]
    right = ((final["occupancy"][:, :20] > 0) == target["occupied"].bool()) | ~known
    count = int(known.sum())
    cell_accuracy = float((right & known).sum()/known.sum()) if count else None
    geometry = geometry_error(final["lattice"], target["lattice"]).norm(dim=-1).mean(-1)
    registered = target["registered"]
    return {"known_cells": count, "cell_accuracy": cell_accuracy,
            "exact_locked_board": bool(right.all()) if count == known.numel() else None,
            "mean_geometry_error_cells": float(geometry[registered].mean()) if bool(registered.any()) else None,
            "note": "Synthetic M2 diagnostics; not complete-state or real-image accuracy."}
