"""Image-supported M2 targets. Unknown values never become negative labels."""
import torch
from torch.nn import functional as F

from .shapes import normalize, rotated_shape
from .state import check

TYPE_NAMES = tuple("IOTSZJLG")
PIECE_NAMES = tuple("IOTSZJL")


def targets_for(state, prepared, *, component_capacity=0, layer_evidence=False):
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
    result = {key: value.unsqueeze(0) for key, value in {
        "lattice": lattice, "registered": torch.tensor(registered),
        "occupied": occupied, "known": known, "semantic": semantic, "typed": typed,
        "layer_type": layer_type, "layer_known": layer_known, "pose_known": pose_known,
        "column": column, "row": row, "rotations": rotations}.items()}
    check(type(layer_evidence) is bool, "layer evidence target option must be boolean")
    if layer_evidence:
        from .layer_evidence import dense_targets
        # Image support is independent of whether locked occupancy is certain.
        support = cell_valid if registered and readable_scale else torch.zeros_like(known)
        result["layer_cells"] = dense_targets(state, result, support[None])
    check(type(component_capacity) is int and 0 <= component_capacity <= 64, "bounded optional component target capacity required")
    if component_capacity:
        # Source Icon fields have no pixel support geometry. Until a resized
        # component observer exists, refuse altered/invalid source evidence.
        check(prepared.source_size == prepared.resized_size and
              bool(prepared.valid[:, :prepared.resized_size[1], :prepared.resized_size[0]].all()),
              "M3 component targets require unchanged fully valid source pixels")
        from .components import component_tensors
        result["components"] = component_tensors([state], capacity=component_capacity)
    return result


def masked_mean(value, mask):
    return (value*mask).sum()/mask.sum().clamp_min(1)


def balanced_occupancy(logits, target, mask, positive_weight=1):
    """Per-image class-balanced cell error; ``positive_weight`` is the class budget.

    The weight multiplies only the occupied-cell mean and the divisor grows with it, so
    the term's total magnitude is unchanged. This moves the per-cell decision; it is not
    another scale on the whole term (that lever is closed).
    """
    errors = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    positive, negative = mask & (target == 1), mask & (target == 0)
    positive_count, negative_count = positive.sum((1, 2)), negative.sum((1, 2))
    per_image = ((positive_weight*(errors*positive).sum((1, 2))/positive_count.clamp_min(1))+
                 (errors*negative).sum((1, 2))/negative_count.clamp_min(1))
    per_image = per_image/(positive_weight*(positive_count > 0).float()+
                           (negative_count > 0).float()).clamp_min(1)
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


# Declared structural scale for the fit-gate lever, and its admission join. One tuple so
# the driver, the loss and the admission gate cannot disagree (V-052).
OCCUPANCY_FACTORS = (1, 4)


def require_occupancy_detail(detail, summary):
    """One declaration of the admission join for the precision detail tap."""
    if (detail not in (0, 1) or
            summary.get("arguments", {}).get("occupancy_detail", 0) != detail or
            summary.get("recovery_contract", {}).get("occupancy_detail", 0) != detail):
        raise ValueError("occupancy detail differs from resource admission")


def require_occupancy_detail_stencil(stencil, summary):
    """One declaration of the admission join for the 2x2 in-cell detail stencil."""
    if (type(stencil) is not int or stencil not in (0, 1) or
            summary.get("arguments", {}).get("occupancy_detail_stencil", 0) != stencil or
            summary.get("recovery_contract", {}).get("occupancy_detail_stencil", 0) != stencil):
        raise ValueError("occupancy detail stencil differs from resource admission")


def require_occupancy_deform(deform, summary):
    """One declaration of the admission join for the per-cell registration lever."""
    if (type(deform) is not int or deform not in (0, 1, 2, 3) or
            summary.get("arguments", {}).get("occupancy_deform", 0) != deform or
            summary.get("recovery_contract", {}).get("occupancy_deform", 0) != deform):
        raise ValueError("occupancy deform differs from resource admission")


# Declared placed-target lever: 0 is the published objective; 1 supervises the displacement the
# detail tap applies against the displacement that would put each cell's sample on the cell it
# should have read, derived from the target lattice. The weight is the auxiliary-term scale the
# semantic, visibility and pose heads already use, declared before any receipt: the term reports
# where a sample belongs and must not be able to dominate the published objective.
OCCUPANCY_PLACED_WEIGHTS = (0, 1)
PLACED_TARGET_WEIGHT = .25


def require_occupancy_placed(value, summary):
    """One declaration of the admission join for the placed-target lever."""
    if (type(value) is not int or value not in OCCUPANCY_PLACED_WEIGHTS or
            summary.get("arguments", {}).get("occupancy_placed", 0) != value or
            summary.get("recovery_contract", {}).get("occupancy_placed", 0) != value):
        raise ValueError("occupancy placed differs from resource admission")


OCCUPANCY_COLUMN_ARMS = (0, 1)
OCCUPANCY_OFFSET_ROWS = (-2, -1, 0, 1, 2)
OCCUPANCY_NORMALISE_ARMS = (0, 1)
OCCUPANCY_MARGIN_HINGE_ARMS = (0, 1)
# Declared before any receipt (docs/M9_SUPERVISED_MARGIN.md), and not searched afterwards: one logit of
# separation between the board's lowest occupied cell and its highest empty cell, at half the weight of
# the occupancy likelihood term (factor 4), so the term shapes the boundary without dominating the fit.
MARGIN_HINGE_MARGIN = 1.0
MARGIN_HINGE_WEIGHT = .5
OCCUPANCY_ANCHOR_ARMS = (0, 1)
# Declared before any receipt (docs/M10_SUPERVISED_ANCHOR.md) and not searched afterwards: the same one-logit
# margin as the hinge, anchored to zero on both sides, at the same half-occupancy weight. The margin is a unit of
# the read's own scale; it is not a threshold and does not move.
ANCHOR_MARGIN = 1.0
ANCHOR_WEIGHT = .5


def require_occupancy_offset_rows(value, summary):
    """One declaration of the admission join for the declared sampling offset."""
    if (type(value) is not int or value not in OCCUPANCY_OFFSET_ROWS or
            summary.get("arguments", {}).get("occupancy_offset_rows", 0) != value or
            summary.get("recovery_contract", {}).get("occupancy_offset_rows", 0) != value):
        raise ValueError("occupancy offset rows differs from resource admission")


def require_occupancy_anchor(value, summary):
    """One declaration of the admission join for the supervised two-sided anchor."""
    if (type(value) is not int or value not in OCCUPANCY_ANCHOR_ARMS or
            summary.get("arguments", {}).get("occupancy_anchor", 0) != value or
            summary.get("recovery_contract", {}).get("occupancy_anchor", 0) != value):
        raise ValueError("occupancy anchor differs from resource admission")


def require_occupancy_margin_hinge(value, summary):
    """One declaration of the admission join for the supervised margin."""
    if (type(value) is not int or value not in OCCUPANCY_MARGIN_HINGE_ARMS or
            summary.get("arguments", {}).get("occupancy_margin_hinge", 0) != value or
            summary.get("recovery_contract", {}).get("occupancy_margin_hinge", 0) != value):
        raise ValueError("occupancy margin hinge differs from resource admission")


def require_occupancy_normalise(value, summary):
    """One declaration of the admission join for the declared output parameterisation."""
    if (type(value) is not int or value not in OCCUPANCY_NORMALISE_ARMS or
            summary.get("arguments", {}).get("occupancy_normalise", 0) != value or
            summary.get("recovery_contract", {}).get("occupancy_normalise", 0) != value):
        raise ValueError("occupancy normalise differs from resource admission")


def require_occupancy_column(value, summary):
    """One declaration of the admission join for the column read-out lever."""
    if (type(value) is not int or value not in OCCUPANCY_COLUMN_ARMS or
            summary.get("arguments", {}).get("occupancy_column", 0) != value or
            summary.get("recovery_contract", {}).get("occupancy_column", 0) != value):
        raise ValueError("occupancy column differs from resource admission")


def placed_target_displacement(lattice, target_lattice, rows, columns, reach):
    """Per-cell displacement, in cells, from where the tap samples to where it should sample.

    The target is the *target lattice's* cell of the same index, so this is information the
    occupancy labels alone cannot carry: it says which cell the tap should have read, not what
    that cell contains. Both latices are detached - the box is supervised by the geometry term,
    and this term must only ever move the field. Targets are clipped to the arm's own reach, so a
    lattice that is further away than the field can move does not ask for an impossible sample.
    """
    device = lattice.device
    row, column = torch.meshgrid(torch.arange(rows, device=device),
                                 torch.arange(columns, device=device), indexing="ij")
    coordinates = torch.stack((column+.5, row+.5), -1).reshape(1, rows*columns, 1, 2).to(lattice.dtype)

    def points(basis):
        return (basis.detach()[:, None, None, 0]+coordinates[..., :1]*basis.detach()[:, None, None, 1]
                +coordinates[..., 1:]*basis.detach()[:, None, None, 2])

    delta = points(target_lattice)-points(lattice)
    basis = lattice.detach()[:, None, 1:]
    u, v = basis[..., 0, :], basis[..., 1, :]
    determinant = u[..., 0]*v[..., 1]-u[..., 1]*v[..., 0]
    determinant = torch.where(determinant.abs() < 1e-6, torch.full_like(determinant, 1e-6),
                              determinant)
    # The basis is applied to the target-minus-predicted *difference*, so these are already the
    # displacement in cells along each axis: no cell index is subtracted (unlike an absolute
    # position, which would carry the cell's own column and row).
    along_u = ((delta[..., 0]*v[..., 1]-delta[..., 1]*v[..., 0])/determinant).squeeze(-1)
    along_v = ((u[..., 0]*delta[..., 1]-u[..., 1]*delta[..., 0])/determinant).squeeze(-1)
    displacement = torch.stack((along_u, along_v), -1)
    return displacement.clamp(-reach, reach)


def placed_displacement_term(output, target, reach=None):
    """Masked smooth-L1 between the applied field and the placed-target displacement."""
    if reach is None:
        from .model import DEFORM_REACH
        reach = DEFORM_REACH
    offsets = output.get("occupancy_deform_offsets")
    if offsets is None:
        raise ValueError("the placed-target term needs the deformation field the tap applied")
    rows, columns = (int(value) for value in target["occupied"].shape[-2:])
    total = int(offsets.shape[1])
    if columns <= 0 or total % columns:
        raise ValueError("the applied field does not cover a rectangular cell grid")
    produced = total//columns
    if produced < rows:
        raise ValueError("the applied field cannot cover the labelled board")
    field = offsets.reshape(offsets.shape[0], produced, columns, 2)[:, :rows]
    field = field.reshape(offsets.shape[0], rows*columns, 2)
    wanted = placed_target_displacement(output["lattice"], target["lattice"], rows, columns, reach)
    if wanted.shape != field.shape:
        raise ValueError("placed-target displacement and applied field differ in shape")
    error = F.smooth_l1_loss(field, wanted, reduction="none").mean(-1)
    known = target["known"][:, :rows].reshape(offsets.shape[0], -1).bool()
    return masked_mean(error, known)


def require_occupancy_resolution(resolution, summary):
    """One declaration of the admission join for the stride-1 resolution lever."""
    if (type(resolution) is not int or resolution not in (0, 1) or
            summary.get("arguments", {}).get("occupancy_resolution", 0) != resolution or
            summary.get("recovery_contract", {}).get("occupancy_resolution", 0) != resolution):
        raise ValueError("occupancy resolution differs from resource admission")


def require_occupancy_context(context, summary):
    """One declaration of the admission join for the occupancy representation lever."""
    if (context not in (0, 1) or
            summary.get("arguments", {}).get("occupancy_context", 0) != context or
            summary.get("recovery_contract", {}).get("occupancy_context", 0) != context):
        raise ValueError("occupancy context differs from resource admission")


def require_occupancy_factor(factor, summary):
    """One declaration of the admission join for the occupancy scale."""
    if (type(factor) is not int or factor not in OCCUPANCY_FACTORS or
            summary.get("arguments", {}).get("occupancy_factor", 1) != factor or
            summary.get("recovery_contract", {}).get("occupancy_factor", 1) != factor):
        raise ValueError("occupancy factor differs from resource admission")


# Declared class budget on occupied cells inside the balanced per-cell term, and its
# admission join. Unlike OCCUPANCY_FACTORS this changes the split between classes, not
# the term's total weight: the closed scale lever is never re-run through this name.
POSITIVE_CELL_WEIGHTS = (1, 4)


def require_positive_weight(weight, summary):
    """One declaration of the admission join for the occupancy class budget."""
    if (type(weight) is not int or weight not in POSITIVE_CELL_WEIGHTS or
            summary.get("arguments", {}).get("occupancy_positive_weight", 1) != weight or
            summary.get("recovery_contract", {}).get("occupancy_positive_weight", 1) != weight):
        raise ValueError("occupancy positive weight differs from resource admission")


# Declared geometry-stage policy: 0 averages every stage's lattice error (published objective),
# 1 spends the geometry term on the final lattice only - the one the occupancy head reads.
GEOMETRY_STAGE_POLICIES = (0, 1)
LATTICE_SUPPORT_LEVERS = (0, 1)
LATTICE_RESIDUAL_LEVERS = (0, 1)
OCCUPANCY_LATTICE_FREE_LEVERS = (0, 1)
LATTICE_BOARD_REFINE_LEVERS = (0, 1)
LATTICE_BOARD_REFINE_LEVERS = (0, 1)


def require_lattice_support(value, summary):
    """One declaration of the admission join for the box-origin support lever."""
    if (type(value) is not int or value not in LATTICE_SUPPORT_LEVERS or
            summary.get("arguments", {}).get("lattice_support", 0) != value or
            summary.get("recovery_contract", {}).get("lattice_support", 0) != value):
        raise ValueError("lattice support differs from resource admission")


# Declared box-extent lever: 0 is the published objective; 1 adds a relative log-scale term on the
# final lattice - the axis-resolved quantity the placement probe reports as L1/L2.
BOX_EXTENT_WEIGHTS = (0.0, 1.0)


def require_box_extent_weight(weight, summary):
    """One declaration of the admission join for the box-extent supervision lever."""
    if (type(weight) is not float or weight not in BOX_EXTENT_WEIGHTS or
            summary.get("arguments", {}).get("box_extent_weight", 0.0) != weight or
            summary.get("recovery_contract", {}).get("box_extent_weight", 0.0) != weight):
        raise ValueError("box extent weight differs from resource admission")


def extent_error(lattice, target):
    """Relative log-scale error of the lattice basis per axis, in nats: shape [batch, 2]."""
    predicted = torch.stack((lattice[:, 1].norm(dim=-1), lattice[:, 2].norm(dim=-1)), -1).clamp_min(1e-6)
    reference = torch.stack((target[:, 1].norm(dim=-1), target[:, 2].norm(dim=-1)), -1).clamp_min(1e-6)
    return torch.log(predicted/reference)


# Declared geometry-capacity lever: 0 is the published objective; 1 adds a boundary support-anchor
# geometry term on the final lattice. The published geometry mean dilutes the four boundary anchors
# into 200 interior centers - they carry 1/51 of it - so the box is weakly pinned exactly where it is
# read. This term spends weight 1.0, the geometry term's own weight, on the eight boundary support
# anchors.
SUPPORT_ANCHOR_WEIGHTS = (0.0, 1.0)
SUPPORT_ANCHORS = ((0., 0.), (5., 0.), (10., 0.), (0., 10.), (10., 10.),
                   (0., 20.), (5., 20.), (10., 20.))


def support_anchor_error(lattice, target):
    """Normalized error of the eight boundary support anchors, shape [batch, 8, 2]."""
    coordinates = torch.tensor(SUPPORT_ANCHORS, device=lattice.device, dtype=lattice.dtype)

    def points(basis):
        return (basis[:, None, 0]+coordinates[None, :, :1]*basis[:, None, 1]
                +coordinates[None, :, 1:]*basis[:, None, 2])
    spacing = torch.stack((target[:, 1].norm(dim=-1), target[:, 2].norm(dim=-1)), -1).clamp_min(1e-6)
    return (points(lattice)-points(target))/spacing[:, None]


def support_anchor_term(lattice, target, registered):
    """The declared capacity term: masked mean smooth-L1 of the boundary anchors' error."""
    error = support_anchor_error(lattice, target)
    return masked_mean(F.smooth_l1_loss(error, torch.zeros_like(error), reduction="none").mean((1, 2)),
                       registered)


def require_occupancy_lattice_free(value, summary):
    """One declaration of the admission join for the lattice-agnostic read-out lever."""
    if (type(value) is not int or value not in OCCUPANCY_LATTICE_FREE_LEVERS or
            summary.get("arguments", {}).get("occupancy_lattice_free", 0) != value or
            summary.get("recovery_contract", {}).get("occupancy_lattice_free", 0) != value):
        raise ValueError("occupancy lattice-free readout differs from resource admission")


def require_lattice_board_refine(value, summary):
    """One declaration of the admission join for the per-board refinement capacity lever."""
    if (type(value) is not int or value not in LATTICE_BOARD_REFINE_LEVERS or
            summary.get("arguments", {}).get("lattice_board_refine", 0) != value or
            summary.get("recovery_contract", {}).get("lattice_board_refine", 0) != value):
        raise ValueError("lattice board refinement differs from resource admission")


def require_lattice_board_refine(value, summary):
    """One declaration of the admission join for the per-board refinement capacity lever."""
    if (type(value) is not int or value not in LATTICE_BOARD_REFINE_LEVERS or
            summary.get("arguments", {}).get("lattice_board_refine", 0) != value or
            summary.get("recovery_contract", {}).get("lattice_board_refine", 0) != value):
        raise ValueError("lattice board refinement differs from resource admission")


def require_lattice_residual(value, summary):
    """One declaration of the admission join for the per-stage lattice residual lever."""
    if (type(value) is not int or value not in LATTICE_RESIDUAL_LEVERS or
            summary.get("arguments", {}).get("lattice_residual", 0) != value or
            summary.get("recovery_contract", {}).get("lattice_residual", 0) != value):
        raise ValueError("lattice residual differs from resource admission")


def require_support_anchor_weight(weight, summary):
    """One declaration of the admission join for the capacity (support-anchor) lever."""
    if (type(weight) is not float or weight not in SUPPORT_ANCHOR_WEIGHTS or
            summary.get("arguments", {}).get("support_anchor_weight", 0.0) != weight or
            summary.get("recovery_contract", {}).get("support_anchor_weight", 0.0) != weight):
        raise ValueError("support anchor weight differs from resource admission")


def require_geometry_final_only(value, summary):
    """One declaration of the admission join for the geometry stage policy."""
    if (type(value) is not int or value not in GEOMETRY_STAGE_POLICIES or
            summary.get("arguments", {}).get("geometry_final_only", 0) != value or
            summary.get("recovery_contract", {}).get("geometry_final_only", 0) != value):
        raise ValueError("geometry stage policy differs from resource admission")


def visible_window(tensor, target, name):
    """The labelled visible board of a per-cell map, sized by the target, refused on mismatch.

    `tensor` is `(batch, visible_rows[+spawn_rows], columns[, classes])` and the target's `occupied` field
    is the labelled `(visible_rows, columns)`. Sizing the window from the label shape is what stops a
    structural change from silently truncating a score; a map that cannot cover the board is an error
    rather than a shorter slice (V-089).
    """
    rows, columns = (int(value) for value in target["occupied"].shape[-2:])
    shape = tuple(int(value) for value in tensor.shape)
    if tensor.dim() == 3:
        produced_rows, produced_columns = shape[1], shape[2]
    elif tensor.dim() == 4:
        produced_rows, produced_columns = shape[1], shape[2]
    else:
        raise ValueError("%s must be (batch, rows, columns[, classes]), got %s" % (name, shape))
    if produced_columns != columns or produced_rows < rows:
        raise ValueError("%s cannot cover the labelled %dx%d board: %s"
                         % (name, rows, columns, shape))
    return tensor[:, :rows]


def margin_hinge_term(output, target):
    """Declared supervision of the decision boundary: per board, the separation that thresholding needs.

    For every board carrying both occupied and empty *known* cells, the hinge is
    `max(0, margin - (lowest_occupied_logit - highest_empty_logit))`: it asks that the two modes not overlap
    by more than the declared margin, and nothing else. It is supervision, not a boundary rule - it never
    names a threshold, never touches the ranking of cells within a mode, and reads only the labels the
    occupancy term already uses (`unknown` cells are excluded from both sides, so an invisible cell is
    never treated as empty). Boards where one side is absent contribute nothing and are not counted, so a
    batch of empty boards has an exactly-zero hinge rather than an undefined one.
    """
    logits = visible_window(output["occupancy"], target, "occupancy")
    occupied = target["occupied"].bool() & target["known"].bool()
    empty = (target["occupied"] == 0) & target["known"].bool()
    total = logits.sum()*0.0
    qualifying = 0
    for index in range(logits.shape[0]):
        high, low = logits[index][occupied[index]], logits[index][empty[index]]
        if high.numel() and low.numel():
            total = total+F.relu(MARGIN_HINGE_MARGIN-(high.min()-low.max()))
            qualifying += 1
    return total/qualifying if qualifying else total


def anchor_term(output, target):
    """Declared supervision of the operating point: the margin, anchored to zero on both sides.

    For every board carrying both occupied and empty *known* visible cells, the term is
    `max(0, margin - lowest_occupied) + max(0, margin + highest_empty)`: the first half asks that every
    occupied cell sit at least `margin` above the `0.0` boundary a consumer applies, the second that every
    empty cell sit at least `margin` below it. The one-sided hinge could be satisfied by separating the two
    modes anywhere on the axis, which is how a better ranker became a worse thresholded detector (M9); this
    term names `0.0` itself, so the operating point is trained rather than inherited. It is supervision, not
    a boundary rule: it chooses no threshold, counts no cells, reads no prevalence, adds no parameter, and
    excludes `unknown` cells from both sides so an invisible cell is never treated as empty.
    """
    logits = visible_window(output["occupancy"], target, "occupancy")
    occupied = target["occupied"].bool() & target["known"].bool()
    empty = (target["occupied"] == 0) & target["known"].bool()
    total = logits.sum()*0.0
    qualifying = 0
    for index in range(logits.shape[0]):
        high, low = logits[index][occupied[index]], logits[index][empty[index]]
        if high.numel() and low.numel():
            total = total+F.relu(ANCHOR_MARGIN-high.min())+F.relu(ANCHOR_MARGIN+low.max())
            qualifying += 1
    return total/qualifying if qualifying else total


def stage_loss(output, target, identity_factor=1, endpoint_term="joint", occupancy_factor=1,
               positive_weight=1, occupancy_placed=0, occupancy_margin_hinge=0, occupancy_anchor=0):
    error = geometry_error(output["lattice"], target["lattice"])
    geometry = masked_mean(F.smooth_l1_loss(error, torch.zeros_like(error), reduction="none").mean((1, 2)), target["registered"])
    occupancy = balanced_occupancy(visible_window(output["occupancy"], target, "occupancy"),
                                   target["occupied"], target["known"], positive_weight)
    type_error = F.cross_entropy(visible_window(output["semantic"], target, "semantic").permute(0, 3, 1, 2),
                                 target["semantic"], reduction="none")
    semantic = masked_mean(type_error, target["typed"])
    visibility = F.binary_cross_entropy_with_logits(visible_window(output["observability"], target,
                                                                 "observability"), target["known"].float())
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
    result = {"geometry": geometry, "occupancy": occupancy, "semantic": semantic,
            "visibility": visibility, "registration": registration, "layers": layers,
            "layer_visibility": layer_visibility, "pose": pose}
    check(("layer_cells" in output) == ("layer_cells" in target), "layer evidence outputs/targets must match")
    if "layer_cells" in output:
        label = target["layer_cells"]
        result["layer_cells"] = balanced_occupancy(output["layer_cells"].flatten(0, 1),
            label["occupied"].flatten(0, 1), label["known"].flatten(0, 1))
    check(type(occupancy_factor) is int and occupancy_factor in OCCUPANCY_FACTORS,
          "occupancy factor must be1 or4")
    check(type(positive_weight) is int and positive_weight in POSITIVE_CELL_WEIGHTS,
          "occupancy positive weight must be1 or4")
    check(type(occupancy_placed) is int and occupancy_placed in OCCUPANCY_PLACED_WEIGHTS,
          "occupancy placed must be0 or1")
    if occupancy_placed:
        # Declared placed-target lever: the term names the cell each sample should have read, which
        # the occupancy label alone cannot express, and it moves only the field.
        result["placed"] = placed_displacement_term(output, target)
    check(type(occupancy_margin_hinge) is int and occupancy_margin_hinge in OCCUPANCY_MARGIN_HINGE_ARMS,
          "occupancy margin hinge must be 0 (unsupervised boundary) or 1 (declared within-board margin)")
    if occupancy_margin_hinge:
        # Declared boundary supervision: the read has never been told where its decision boundary should
        # be, so this asks for the separation a threshold needs without naming the threshold.
        result["margin"] = margin_hinge_term(output, target)
    check(type(occupancy_anchor) is int and occupancy_anchor in OCCUPANCY_ANCHOR_ARMS,
          "occupancy anchor must be 0 (unsupervised boundary) or 1 (declared two-sided margin at zero)")
    if occupancy_anchor:
        # Declared operating-point supervision: the margin is anchored to the boundary a consumer applies,
        # so pinning the location is part of the objective rather than a choice made afterwards.
        result["anchor"] = anchor_term(output, target)
    check(("components" in output) == ("components" in target), "joint component outputs/targets must match")
    if "components" in output:
        from .components import component_loss
        result["components"] = component_loss(output["components"], target["components"],
                                              identity_factor=identity_factor, endpoint_term=endpoint_term)[0]
    return result


def joint_loss(outputs, target, identity_factor=1, endpoint_term="joint", occupancy_factor=1,
               positive_weight=1, geometry_final_only=0, box_extent_weight=0.0,
               support_anchor_weight=0.0, occupancy_placed=0, occupancy_margin_hinge=0,
               occupancy_anchor=0):
    check(type(geometry_final_only) is int and geometry_final_only in GEOMETRY_STAGE_POLICIES,
          "geometry stage policy must be0 or1")
    check(type(box_extent_weight) is float and box_extent_weight in BOX_EXTENT_WEIGHTS,
          "box extent weight must be0 or1")
    check(type(support_anchor_weight) is float and support_anchor_weight in SUPPORT_ANCHOR_WEIGHTS,
          "support anchor weight must be0 or1")
    per_stage = [stage_loss(stage, target, identity_factor, endpoint_term, occupancy_factor,
                            positive_weight, occupancy_placed, occupancy_margin_hinge, occupancy_anchor)
                 for stage in outputs["stages"]]
    terms = {key: sum(stage[key] for stage in per_stage)/len(per_stage) for key in per_stage[0]}
    if geometry_final_only:
        # Declared geometry lever: the occupancy read-out reads the final lattice only, so
        # every other stage's geometry mean is replaced by that one rather than diluting it.
        terms["geometry"] = per_stage[-1]["geometry"]
    if box_extent_weight:
        # Declared extent lever: the final basis's relative log-scale error, axis-resolved. The origin
        # and the occupancy read-out are untouched, so this is supervision of the four box parameters'
        # scale rather than of where the lattice sits.
        terms["box_extent"] = masked_mean(
            extent_error(outputs["final"]["lattice"], target["lattice"]).abs().mean(-1),
            target["registered"])
    if support_anchor_weight:
        # Declared capacity lever: the final lattice's boundary support anchors, supervised on
        # their own instead of diluted into the 200 interior centers.
        terms["support_anchor"] = support_anchor_term(outputs["final"]["lattice"],
                                                     target["lattice"], target["registered"])
    # Pilot hypotheses, not tuned or established research hyperparameters.
    weights = {"geometry": 1., "occupancy": 1.*occupancy_factor, "semantic": .25, "visibility": .25,
               "registration": .25, "layers": .5, "layer_visibility": .25, "pose": .25,
               "components": .5, "layer_cells": .5, "box_extent": box_extent_weight,
               "support_anchor": support_anchor_weight,
               "placed": PLACED_TARGET_WEIGHT*occupancy_placed,
               "margin": MARGIN_HINGE_WEIGHT*occupancy_margin_hinge,
               "anchor": ANCHOR_WEIGHT*occupancy_anchor}
    total = sum(weights[key]*value for key, value in terms.items())
    return total, terms


@torch.no_grad()
def diagnostic_metrics(outputs, target):
    final = outputs["final"]
    known = target["known"]
    right = ((visible_window(final["occupancy"], target, "occupancy") > 0) == target["occupied"].bool()) | ~known
    count = int(known.sum())
    cell_accuracy = float((right & known).sum()/known.sum()) if count else None
    geometry = geometry_error(final["lattice"], target["lattice"]).norm(dim=-1).mean(-1)
    registered = target["registered"]
    result = {"known_cells": count, "cell_accuracy": cell_accuracy,
            "exact_locked_board": bool(right.all()) if count == known.numel() else None,
            "mean_geometry_error_cells": float(geometry[registered].mean()) if bool(registered.any()) else None,
            "note": "Synthetic M2 diagnostics; not complete-state or real-image accuracy."}
    if "layer_cells" in final:
        from .layer_evidence import grid_metrics
        result.update(grid_metrics(final["layer_cells"], target["layer_cells"], target["layer_type"]))
    return result
