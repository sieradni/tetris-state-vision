"""Finite, source-frozen CPU curriculum. Resource profile required before a study."""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

from accumulation_checkpoint import create_journal, make_contract, restore_boundary
from accumulation_contract import POLICIES as ACCUMULATION_POLICIES, group_specs, update_count
from probe_model import bounded_job_snapshot, require_bounded_job
from source_snapshot import capture_source, verify_source
from atomic_write import replace_retry

ROOT = Path(__file__).resolve().parents[1]
NATIVE_CACHE_LIMITS = {"ONEDNN_PRIMITIVE_CACHE_CAPACITY": "16", "DNNL_PRIMITIVE_CACHE_CAPACITY": "16"}
# Mirrors tetris_vision.components.ENDPOINT_TERMS without importing torch at parser build
# time; a test binds the two so the CLI and the admission gate cannot drift apart (V-052).
ENDPOINT_TERMS = ("joint", "separate", "length")
# Mirrors tetris_vision.losses.OCCUPANCY_FACTORS without importing torch at parser build
# time; a test binds the two (V-052 pattern).
OCCUPANCY_FACTORS = (1, 4)
# Mirrors tetris_vision.losses.POSITIVE_CELL_WEIGHTS; the same test binds the pair.
POSITIVE_CELL_WEIGHTS = (1, 4)
# Mirrors tetris_vision.losses.OCCUPANCY_PLACED_WEIGHTS; the same test binds the pair.
OCCUPANCY_PLACED_WEIGHTS = (0, 1)
# Mirrors tetris_vision.losses.OCCUPANCY_OFFSET_ROWS; the same test binds the pair.
OCCUPANCY_OFFSET_ROWS = (-2, -1, 0, 1, 2)

# Mirrors tetris_vision.losses.BOX_EXTENT_WEIGHTS; the same test binds the pair (V-052 pattern).
BOX_EXTENT_WEIGHTS = (0.0, 1.0)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=("pilot", "accuracy"), required=True)
    parser.add_argument("--component-capacity", type=int, default=0,
                        help="0 keeps M2;10..64 enables joint heads and final-capture data")
    parser.add_argument("--component-identity-factor", type=int, choices=(1, 4), default=1,
                        help="declared queue-token/hold-piece loss scale;1 is the published objective")
    parser.add_argument("--component-identity-readout", choices=("separate", "shared", "prototype", "positional"),
                        default="separate",
                        help="shared reads identity through the cell-type head;prototype reads per-piece image "
                             "queries;positional adds per-position image queries;separate is published")
    parser.add_argument("--component-endpoint-term", choices=ENDPOINT_TERMS, default="joint",
                        help="joint keeps the endpoint as one class of the piece softmax;joint is published;"
                             "separate gives the endpoint its own binary decision;length decides the queue length "
                             "once per case")
    parser.add_argument("--occupancy-factor", type=int, choices=OCCUPANCY_FACTORS, default=1,
                        help="declared locked-occupancy loss scale for the fit-gate lever;1 is published")
    parser.add_argument("--occupancy-positive-weight", type=int, choices=POSITIVE_CELL_WEIGHTS, default=1,
                        help="declared class budget on occupied cells inside the balanced occupancy term;"
                             "1 is published and leaves the term magnitude unchanged")
    parser.add_argument("--occupancy-pixel-query", type=int, choices=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), default=0,
                        help="declared per-cell pixel-statistics read-out (M28): deterministic colour/contrast"
                             " features at the model's own lattice through a zero-initialized projection")
    parser.add_argument("--pixel-query-level", type=int, choices=(0, 1), default=0,
                        help="declared M30 level freedom on the pixel projection: a strictly positive "
                             "scale (1+tanh) and a bias, identity at step zero")
    parser.add_argument("--occupancy-read-standardise", type=int, choices=(0, 1), default=0,
                        help="declared M30 label-free monotone per-board standardisation of the read")
    parser.add_argument("--occupancy-mass", type=int, choices=(0, 1), default=0,
                        help="declared M32 distribution shaping: each board's predicted mass matches its own occupancy")
    parser.add_argument("--occupancy-read-level", type=int, choices=(0, 1), default=0,
                        help="declared M31 mechanism change: the board's own standardised read is the read")
    parser.add_argument("--occupancy-balance", type=int, choices=(0, 1), default=0,
                        help="declared M30 data-derived class budget in the occupancy loss")
    parser.add_argument("--occupancy-context", type=int, choices=(0, 1), default=0,
                        help="declared occupancy representation lever;1 gives the read-out its eight "
                             "lattice neighbors through a zero-initialized residual block;0 is published")
    parser.add_argument("--occupancy-detail", type=int, choices=(0, 1), default=0,
                        help="declared cell-precision lever;1 gives the occupancy read-out the stride-2 "
                             "first-stage feature sampled at each cell center through a zero-initialized "
                             "projection;0 is published")
    parser.add_argument("--occupancy-detail-stencil", type=int, choices=(0, 1), default=0,
                        help="declared stencil form of the cell-precision tap;1 reads the four "
                             "quarter-cell samples of a 2x2 in-cell grid instead of the single center "
                             "sample through the same zero-initialized projection; requires --occupancy-detail 1")
    parser.add_argument("--occupancy-tap-span", type=int, choices=(0, 1), default=0,
                        help="declared span form of the cell-precision tap;1 reads three samples along the "
                             "board's row axis - the cell's own row and the rows immediately above and below "
                             "it - through the same zero-initialized projection, so the read can express the "
                             "row-to-row change that marks a stack boundary; requires --occupancy-detail 1")
    parser.add_argument("--occupancy-resolution", type=int, choices=(0, 1), default=0,
                        help="declared resolution lever;1 adds a stride-1 conv of the image whose "
                             "cell-center sample joins the same zero-initialized detail projection, "
                             "so the tap reads pixel resolution; requires --occupancy-detail 1")
    parser.add_argument("--occupancy-deform", type=int, choices=(0, 1, 2, 3), default=0,
                        help="declared registration lever;1 adds a zero-initialized per-cell field, "
                             "predicted from the feature the detail tap reads, that moves that cell's "
                             "sample by at most one cell, so local misalignment is corrected without "
                             "moving the decoded lattice;2 adds a coarse level predicted from an "
                             "eight-times pooled map (up to three cells) on top so the reach covers "
                             "the measured 2.57-cell placement residual;3 de-means that field across the "
                             "board, so its constant mode is exactly zero and only per-cell "
                             "displacement remains expressible; requires --occupancy-detail 1")
    parser.add_argument("--occupancy-offset-rows", type=int, choices=OCCUPANCY_OFFSET_ROWS,
                        default=0,
                        help="declared sampling geometry; a constant in [-2, 2] cell rows (rows "
                             "bottom-first) added to every detail-tap offset, so +1 samples one row "
                             "above the decoded cell and -1 one row below. It is not learned and adds "
                             "no tensor, and 0 is exactly the published function. Requires "
                             "--occupancy-detail 1")
    parser.add_argument("--occupancy-margin-hinge", type=int, choices=(0, 1), default=0,
                        help="declared supervision of the decision boundary;1 adds a within-board hinge "
                             "max(0, margin - (lowest occupied logit - highest empty logit)) over the "
                             "known visible cells of every board that carries both classes, at the "
                             "declared margin and weight. It names no threshold, reads no prevalence and "
                             "adds no parameter, and 0 is exactly the published objective")
    parser.add_argument("--occupancy-anchor", type=int, choices=(0, 1), default=0,
                        help="declared operating-point supervision;1 adds the two-sided anchor "
                             "max(0, margin - lowest occupied logit) + max(0, margin + highest empty logit) "
                             "over the known visible cells of every board that carries both classes, at the "
                             "declared margin and weight. It names no threshold, reads no prevalence and "
                             "adds no parameter, and 0 is exactly the published objective")
    parser.add_argument("--occupancy-quantile-anchor", type=int, choices=(0, 1), default=0,
                        help="declared mass supervision of the operating point;1 adds the anchored margin "
                             "max(0, margin - q10 of the board's occupied logits) + "
                             "max(0, margin + q90 of its empty logits), averaged over the boards that carry "
                             "both classes, at the same declared margin and weight as --occupancy-anchor. It "
                             "names no threshold, counts no cells, reads no prevalence and adds no "
                             "parameter, and 0 is exactly the published objective")
    parser.add_argument("--occupancy-dense-hinge", type=int, choices=(0, 1), default=0,
                        help="declared dense supervision of the operating point;1 charges every known visible cell "
                             "inside the margin band, class-balanced: 0.5 mean of relu(margin - logit) over the "
                             "occupied cells plus 0.5 mean of relu(margin + logit) over the empty cells, averaged "
                             "over the boards that carry both classes, at the declared margin and weight. Equal class "
                             "weight, never a cell count; it names no threshold, reads no prevalence and adds no "
                             "parameter, and 0 is exactly the published objective")
    parser.add_argument("--occupancy-midpoint", type=int, choices=(0, 1), default=0,
                        help="declared location supervision of the operating point;1 charges, per board carrying "
                             "both classes, the distance of the midpoint of its known-cell class means from the "
                             "declared target location 0.0: |(mean_occupied + mean_empty)/2|, averaged over boards "
                             "at the declared weight. Equal class weight, no margin, no threshold, no prevalence "
                             "and no parameter, and 0 is exactly the published objective")
    parser.add_argument("--occupancy-centre", type=int, choices=(0, 1), default=0,
                        help="declared structural option;1 subtracts each board's own mean over its visible "
                             "window from that board's occupancy logits inside the head, before the objective or "
                             "any scorer sees them, so the read has no additive mode left to drift. No label, no "
                             "threshold, no fitted parameter and no prevalence is read, the spread is untouched "
                             "so relative cell structure is unchanged, and 0 is exactly the published function")
    parser.add_argument("--occupancy-stack-monotone", type=int, choices=(0, 1), default=0,
                        help="declared structural option;1 replaces each column of the occupancy logits with its "
                             "monotone envelope from the floor upward (suffix maximum in bottom-first order), so the "
                             "column crosses 0.0 at most once and that crossing is its height. No label, threshold, "
                             "fitted parameter or prevalence is read, monotonicity is the domain's own constraint on a "
                             "locked stack, spawn rows pass through untouched, and 0 is exactly the published function")
    parser.add_argument("--occupancy-crossing-threshold", type=int, choices=(0, 1), default=0,
                        help="declared label-free decision rule;1 subtracts, per board, the exact Otsu level of that "
                             "board's own visible occupancy logits inside the head, so every downstream consumer "
                             "thresholding at 0.0 decides at the board's own crossing instead of a fixed number. It "
                             "reads no label, no prevalence, no threshold grid and no fitted parameter, is applied "
                             "after the envelope and after centring when they are on, and 0 is exactly the published "
                             "function")
    parser.add_argument("--occupancy-skyline-recovery", type=int, choices=(0, 1), default=0,
                        help="declared joint decision;1 recovers each board's whole skyline with a dynamic programme "
                             "over per-column heights that maximises the read's own separation minus a fixed one-logit "
                             "penalty on neighbouring height changes, applied last (after the envelope, centring and "
                             "the crossing rule). It reads no label, no prevalence and no swept constant, is "
                             "deterministic, and 0 is exactly the published per-column decision")
    parser.add_argument("--occupancy-normalise", type=int, choices=(0, 1), default=0,
                        help="declared output parameterisation;1 z-scores the occupancy logits over "
                             "each board's own visible cells (mean 0, unit spread, no label and no "
                             "cross-board statistic) *inside* the model, so the loss and every scorer "
                             "see the same tensor and the 0.0 boundary is the operating point the head "
                             "is trained to place. It adds no parameter and no tensor, and 0 is "
                             "exactly the published function")
    parser.add_argument("--occupancy-column", type=int, choices=(0, 1), default=0,
                        help="declared read-out lever;1 adds the cell's own column, summarised below "
                             "and above it from the decoder's cell features, through a "
                             "zero-initialized projection - so step zero is the published function, "
                             "no sample moves, and the material an occupancy label actually reports "
                             "sits beside the cell instead of being bought as a displacement")
    parser.add_argument("--occupancy-placed", type=int, choices=OCCUPANCY_PLACED_WEIGHTS, default=0,
                        help="declared placed-target lever;1 adds a term that supervises the "
                             "displacement the detail tap applies against the displacement that would "
                             "put each cell's sample on the cell it should have read, derived from the "
                             "target lattice - information the occupancy label alone cannot carry; it "
                             "moves only the field, adds no tensors, and requires --occupancy-deform 2 "
                             "and --occupancy-detail 1")
    parser.add_argument("--geometry-final-only", type=int, choices=(0, 1), default=0,
                        help="declared geometry lever;0 averages every decoder stage's lattice error "
                             "(published objective);1 spends the geometry term on the final lattice "
                             "only, the one the occupancy read-out reads")
    parser.add_argument("--occupancy-free-stencil", type=int, choices=(0, 1, 2, 3), default=0,
                        help="declared detail lever on the free channel;0 reads image-level context and "
                             "fixed cell coordinates only;1 adds a coarse five-tap stencil sampled on the "
                             "fixed lattice, so a cell sees its own coarse box and its four neighbours "
                             "without touching the predicted lattice;2 is the same stencil de-meaned "
                             "across the board;3 reads the same five taps on the locally normalised "
                             "field, so a board-wide offset or rescaling leaves it unchanged")
    parser.add_argument("--occupancy-pair-rank", type=int, choices=(0, 1), default=0,
                        help="declare the within-board pair-order term (M25); 1 requires --occupancy-lattice-free 2")
    parser.add_argument("--occupancy-pair-gap", type=int, choices=(0, 1), default=0,
                        help="declare the symmetric within-board pair margin (M26); 1 requires --occupancy-lattice-free 2")
    parser.add_argument("--occupancy-lattice-free", type=int, choices=(0, 1, 2), default=0,
                        help="declared read-out lever;0 is the published lattice-sampled occupancy "
                             "read-out;1 adds a gated channel that reads image-level context and fixed "
                             "cell coordinates only, so it never samples the predicted lattice")
    parser.add_argument("--lattice-board-refine", type=int, choices=(0, 1), default=0,
                        help="declared capacity lever;0 is the published shared affine decode;1 "
                             "adds one zero-initialized board-conditioned correction of the decoded "
                             "lattice in cell space, applied identically at every stage")
    parser.add_argument("--lattice-residual", type=int, choices=(0, 1), default=0,
                        help="declared capacity lever;0 is the published shared affine decode;1 adds a "
                             "zero-initialized per-stage residual in decoded cell space, so each "
                             "stage's lattice can be corrected without moving the shared box")
    parser.add_argument("--lattice-support", type=int, choices=(0, 1), default=0,
                        help="declared box-origin lever;0 is the published global-token box head;"
                             "1 adds a zero-initialized residual from the coarse feature map's "
                             "spatial moments, so the origin can be board-dependent")
    parser.add_argument("--support-anchor-weight", type=float, default=0.0,
                        help="declared capacity lever: weight of the boundary support-anchor geometry term")
    parser.add_argument("--box-extent-weight", type=float, choices=(0.0, 1.0), default=0.0,
                        help="declared box-extent lever;0 is the published objective;1 adds a "
                             "relative log-scale term on the final lattice, the quantity the "
                             "placement probe measures")
    parser.add_argument("--layer-evidence", action="store_true", help="optional independent visible-cell layer supervision/readout")
    parser.add_argument("--curriculum", choices=("stream-v1", "all-class-fit-v1", "all-class-interleaved-v1"), default="stream-v1")
    parser.add_argument("--capture-max-side", type=int, default=512,
                        help="model capture side;512 reproduces the published representation")
    parser.add_argument("--trunk-dimension", type=int, default=192,
                        help="declared decision-path width;192 is published;240 is the declared wider identity "
                             "(heads 6 requires a multiple of 6, the config refuses anything else)")
    parser.add_argument("--accumulation-policy", choices=tuple(ACCUMULATION_POLICIES),
                        help="whole-group accumulation arm from the predeclared matched comparison")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--lr-final", type=float,
                        help="optional cosine decay from --learning-rate to this rate across the segment's steps; absent keeps the published constant rate")
    parser.add_argument("--type-head-rate-factor", type=int, choices=(1, 4), default=1)
    parser.add_argument("--wall-seconds", type=float, default=300)
    parser.add_argument("--evaluate-every", type=int, default=128)
    parser.add_argument("--development-groups", type=int, default=6)
    parser.add_argument("--profile-square", action="store_true")
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=64)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--max-new-steps", type=int)
    parser.add_argument("--init-weights-from", type=Path,
                        help="owned run whose saved inference weights are this run's recorded initial state")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="staged refinement: train only the shared type head")
    args = parser.parse_args(argv)
    if not 1 <= args.steps <= 4096 or not 0 <= args.seed <= 2**31-1:
        parser.error("steps 1..4096 and finite nonnegative seed required")
    if args.component_capacity != 0 and not 10 <= args.component_capacity <= 64:
        parser.error("joint stream component capacity must be10..64 to cover the renderer's10 slots;0 keeps M2")
    if not 64 <= args.capture_max_side <= 1536:
        parser.error("capture side64..1536 required")
    if args.component_identity_factor != 1 and not args.component_capacity:
        parser.error("component identity factor requires the joint heads")
    if args.component_identity_readout != "separate" and not args.component_capacity:
        parser.error("a declared identity readout other than separate requires the joint heads")
    if args.component_endpoint_term != "joint" and (not args.component_capacity or
                                                    args.component_identity_readout not in ("prototype", "positional")):
        parser.error("a declared endpoint term requires the joint heads and a declared identity readout")
    if args.component_endpoint_term == "length" and args.component_identity_readout != "positional":
        parser.error("a length objective requires the positional identity readout")
    if args.occupancy_factor != 1 and not args.component_capacity:
        parser.error("the occupancy fit lever requires the joint heads")
    if args.occupancy_positive_weight != 1 and not args.component_capacity:
        parser.error("the per-cell occupancy lever requires the joint heads")
    if args.curriculum in ("all-class-fit-v1", "all-class-interleaved-v1") and args.component_capacity:
        parser.error("all-class fit is a capacity0 diagnostic")
    if args.type_head_rate_factor != 1 and (args.curriculum != "all-class-fit-v1" or args.component_capacity or args.layer_evidence or args.learning_rate != .001):
        parser.error("type-head rate intervention requires the plain ordered M2 fit")
    if args.freeze_backbone and (args.init_weights_from is None or args.curriculum != "all-class-fit-v1"):
        parser.error("frozen-backbone refinement needs --init-weights-from and the ordered fit")
    if args.freeze_backbone and (args.component_capacity or args.layer_evidence or args.accumulation_policy is not None):
        parser.error("frozen-backbone refinement is plain ordered M2 only")
    if args.occupancy_deform and not args.occupancy_detail:
        parser.error("the registration lever moves the detail tap's sample; it requires --occupancy-detail 1")
    if args.occupancy_placed and args.occupancy_deform != 2:
        parser.error("the placed-target term supervises the widened field;1 requires --occupancy-deform 2")
    if args.occupancy_tap_span and not args.occupancy_detail:
        parser.error("the tap span is a form of the detail tap's geometry; it requires --occupancy-detail 1")
    if args.occupancy_tap_span and (args.occupancy_detail_stencil or args.occupancy_resolution or
                                    args.occupancy_deform or args.occupancy_offset_rows or
                                    args.occupancy_placed or args.occupancy_context or args.occupancy_column or
                                    args.occupancy_lattice_free or args.occupancy_normalise or
                                    args.occupancy_dense_hinge or args.occupancy_quantile_anchor or
                                    args.occupancy_anchor or args.occupancy_margin_hinge or
                                    args.occupancy_midpoint or
                                    args.occupancy_stack_monotone or
                                    args.occupancy_skyline_recovery):
        # The span is one declared reading geometry: another form of the same tap, a lever that moves the sample, a
        # decision that replaces the read's structure, or a supervised read-out would all leave the
        # row-neighbourhood read confounded with a second change. The two level transforms are the one declared
        # composition (M19), admitted together and only together by the check below.
        parser.error("the tap span is one declared reading geometry, so it is refused beside every other form "
                     "of the tap, every lever that moves the sample, every decision that replaces the read's "
                     "structure, and every supervised read-out")
    if args.occupancy_tap_span and args.occupancy_centre != args.occupancy_crossing_threshold:
        # M19 declares the span under the level transforms: a half composition (one transform without the
        # other) is not a declared arm and would not answer the question that arm exists to answer.
        parser.error("the span's level transforms are declared together: the composition requires both "
                     "--occupancy-centre 1 and --occupancy-crossing-threshold 1, or neither")
    if args.occupancy_offset_rows and not args.occupancy_detail:
        parser.error("the sampling offset moves the detail tap's sample; it requires --occupancy-detail 1")
    if args.occupancy_offset_rows and (args.occupancy_detail_stencil or args.occupancy_resolution or
                                       args.occupancy_context or args.occupancy_deform or
                                       args.occupancy_placed or args.occupancy_column):
        parser.error("the declared sampling offset is a single geometry, not a form of another lever")
    if args.occupancy_skyline_recovery and (args.occupancy_normalise or args.occupancy_dense_hinge or
                                            args.occupancy_quantile_anchor or args.occupancy_anchor or
                                            args.occupancy_margin_hinge or args.occupancy_midpoint or
                                            args.occupancy_context or args.occupancy_detail_stencil or
                                            args.occupancy_resolution or args.occupancy_deform or
                                            args.occupancy_placed or args.occupancy_column or
                                            args.occupancy_offset_rows or args.occupancy_lattice_free):
        # The recovery decides the whole board's structure last, so every term that charges a shift or a margin is
        # moot, the normalised output would redefine the emission, and every read-moving lever would leave the decided
        # skyline confounded with a changed sample. The three structural read options compose with it by declaration.
        parser.error("the skyline recovery decides the board's structure jointly, so the normalised output, the five "
                     "supervision terms and every lever that changes what is read, where it is sampled or where zero "
                     "sits are all refused beside it")
    if args.occupancy_crossing_threshold and (args.occupancy_normalise or args.occupancy_dense_hinge or
                                              args.occupancy_quantile_anchor or args.occupancy_anchor or
                                              args.occupancy_margin_hinge or args.occupancy_midpoint or
                                              args.occupancy_context or args.occupancy_detail_stencil or
                                              args.occupancy_resolution or args.occupancy_deform or
                                              args.occupancy_placed or args.occupancy_column or
                                              args.occupancy_offset_rows or args.occupancy_lattice_free == 1):
        # The rule decides where the read already separates, so every term that charges a shift or a margin is moot,
        # the normalised output would fold the board's spread into the level, and every read-moving lever would leave
        # the selected crossing confounded with a changed sample. The structural read options compose with it by
        # declaration: the envelope places the crossing's shape, centring its level, and the summed two-path read
        # (occupancy_lattice_free 2, M21) forms the score the rule decides - none of them is a sample or a shift.
        parser.error("the crossing threshold decides at the read's own Otsu split, so the normalised output, the five "
                     "supervision terms and every lever that changes what is read, where it is sampled or where zero "
                     "sits are all refused beside it")
    if args.occupancy_stack_monotone and (args.occupancy_normalise or args.occupancy_dense_hinge or
                                         args.occupancy_quantile_anchor or args.occupancy_anchor or
                                         args.occupancy_margin_hinge or args.occupancy_midpoint or
                                         args.occupancy_context or args.occupancy_detail_stencil or
                                         args.occupancy_resolution or args.occupancy_deform or
                                         args.occupancy_placed or args.occupancy_column or
                                         args.occupancy_offset_rows or args.occupancy_lattice_free == 1):
        # The monotone read places the crossing by structure, so every term that charges a shift or a margin is moot
        # on it, the normalised output would replace that crossing with a standard deviation, and every read-moving
        # lever would leave the envelope confounded with a changed sample. Two compositions are declared on it:
        # centring (M14's follow-on) and the summed two-path read (M20, occupancy_lattice_free 2); the single free
        # channel (1) stays its own arm and is refused here.
        parser.error("the stack-contiguous read places the crossing by column structure, so the normalised output, "
                     "the five supervision terms and every lever that changes what is read, where it is sampled or "
                     "where zero sits are all refused beside it")
    if args.occupancy_pixel_query and (args.occupancy_context or args.occupancy_detail or
                                       args.occupancy_detail_stencil or args.occupancy_resolution or
                                       args.occupancy_deform or args.occupancy_column or
                                       args.occupancy_lattice_free or args.occupancy_offset_rows or
                                       args.occupancy_normalise or args.occupancy_centre or
                                       args.occupancy_stack_monotone or args.occupancy_crossing_threshold or
                                       args.occupancy_skyline_recovery or args.occupancy_tap_span or
                                       args.occupancy_placed):
        parser.error("the pixel-statistics read-out is a standalone lever: it is declared alone, so no other "
                     "occupancy read-out selector is set beside it")
    if args.occupancy_normalise and (args.occupancy_context or args.occupancy_detail_stencil or
                                     args.occupancy_resolution or args.occupancy_deform or
                                     args.occupancy_placed or args.occupancy_column or
                                     args.occupancy_offset_rows or args.occupancy_lattice_free):
        parser.error("the normalised output is declared on the published occupancy read-out, so no "
                     "lever that changes what is read or where it is sampled may sit beside it")
    if args.occupancy_centre and (args.occupancy_normalise or args.occupancy_dense_hinge or
                                  args.occupancy_quantile_anchor or args.occupancy_anchor or
                                  args.occupancy_margin_hinge or args.occupancy_midpoint or
                                  args.occupancy_context or args.occupancy_detail_stencil or
                                  args.occupancy_resolution or args.occupancy_deform or
                                  args.occupancy_placed or args.occupancy_column or
                                  args.occupancy_offset_rows or args.occupancy_lattice_free):
        # Structural centring removes the read's additive mode, so every term that charges a shift is moot on it,
        # the normalised output already owns the parameterisation, and every read-moving lever would leave the
        # centred read confounded with a changed sample or a re-parameterised output.
        parser.error("the centred read is a structural option on the published occupancy read-out, so the "
                     "normalised output, the five supervision terms and every lever that changes what is read, "
                     "where it is sampled or where zero sits are all refused beside it")
    if args.occupancy_midpoint and (args.occupancy_dense_hinge or args.occupancy_quantile_anchor or
                                    args.occupancy_anchor or args.occupancy_margin_hinge or
                                    args.occupancy_normalise or args.occupancy_context or
                                    args.occupancy_detail_stencil or args.occupancy_resolution or
                                    args.occupancy_deform or args.occupancy_placed or args.occupancy_column or
                                    args.occupancy_offset_rows or args.occupancy_lattice_free):
        # Location supervision: it supersedes all four existing terms (stacking would train several statements about
        # one boundary) and refuses every lever that changes what is read, where it is sampled or where zero sits,
        # so the supervised location cannot be confounded with a moved read.
        parser.error("the midpoint anchor supervises the published occupancy read-out and supersedes the dense "
                     "hinge, the quantile anchor, the two-sided anchor and the one-sided hinge, so none of them "
                     "nor any lever that changes what is read, where it is sampled or where zero sits may sit "
                     "beside it")
    if args.occupancy_dense_hinge and (args.occupancy_quantile_anchor or args.occupancy_anchor or
                                       args.occupancy_margin_hinge or args.occupancy_normalise or
                                       args.occupancy_context or args.occupancy_detail_stencil or
                                       args.occupancy_resolution or args.occupancy_deform or
                                       args.occupancy_placed or args.occupancy_column or
                                       args.occupancy_offset_rows or args.occupancy_lattice_free):
        # Dense per-cell supervision of the same margin: it supersedes all three per-board summaries (stacking
        # would train several statements about one boundary) and refuses every lever that changes what is read,
        # where it is sampled or where zero sits, so the charged band cannot be confounded with a moved read.
        parser.error("the dense hinge supervises the published occupancy read-out cell by cell and supersedes "
                     "the quantile anchor, the two-sided anchor and the one-sided hinge, so none of them nor any "
                     "lever that changes what is read, where it is sampled or where zero sits may sit beside it")
    if args.occupancy_quantile_anchor and (args.occupancy_anchor or args.occupancy_margin_hinge or
                                           args.occupancy_normalise or args.occupancy_context or
                                           args.occupancy_detail_stencil or args.occupancy_resolution or
                                           args.occupancy_deform or args.occupancy_placed or
                                           args.occupancy_column or args.occupancy_offset_rows or
                                           args.occupancy_lattice_free):
        # Mass supervision of the same margin: it supersedes both extremal terms (stacking would train three
        # statements about one boundary) and refuses every lever that changes what is read, where it is sampled
        # or where zero sits, so the supervised bulk cannot be confounded with a moved read.
        parser.error("the quantile anchor supervises the published occupancy read-out and supersedes the "
                     "two-sided anchor and the one-sided hinge, so neither of them nor any lever that changes "
                     "what is read, where it is sampled or where zero sits may sit beside it")
    if args.occupancy_anchor and (args.occupancy_margin_hinge or args.occupancy_normalise or
                                  args.occupancy_context or args.occupancy_detail_stencil or
                                  args.occupancy_resolution or args.occupancy_deform or
                                  args.occupancy_placed or args.occupancy_column or
                                  args.occupancy_offset_rows or args.occupancy_lattice_free):
        # The anchor is the same family as the one-sided hinge and supersedes it: stacking both would train two
        # margins against one boundary, and every other lever would leave the supervised location confounded
        # with a changed read, a moved sample or a re-parameterised output.
        parser.error("the two-sided anchor supervises the published occupancy read-out and supersedes the "
                     "one-sided hinge, so neither it nor any lever that changes what is read, where it is "
                     "sampled or where zero sits may sit beside it")
    if args.occupancy_margin_hinge and (args.occupancy_normalise or args.occupancy_context or
                                        args.occupancy_detail_stencil or
                                        args.occupancy_resolution or args.occupancy_deform or
                                        args.occupancy_placed or args.occupancy_column or
                                        args.occupancy_offset_rows or
                                        args.occupancy_lattice_free):
        # The margin supervises the published read's boundary. A lever that moves the sample or changes
        # the read form would leave the supervised boundary confounded with it, and the normalised
        # output would put zero at the median before the hinge could answer for it (M8).
        parser.error("the margin hinge supervises the published occupancy read-out, so no lever that "
                     "changes what is read, where it is sampled or where zero sits may sit beside it")
    if args.occupancy_column and (args.occupancy_detail_stencil or args.occupancy_resolution or
                                  args.occupancy_context or args.occupancy_deform or
                                  args.occupancy_placed):
        # The column read-out adds context around the cell, so it may sit beside the detail tap that
        # every arm of its family shares; it is refused with the levers that change *what* is read
        # (stencil, resolution, neighborhood) or move it (deform, placed), because those would leave
        # the read-out confounded with a sampling change.
        parser.error("the column read-out is declared on the published occupancy read-out")
    if args.init_weights_from is not None and args.resume_from is not None and not args.freeze_backbone:
        parser.error("initial weights define a fresh start; resume an existing run instead")
    if not math.isfinite(args.wall_seconds) or not 0 < args.wall_seconds <= 1800 or not math.isfinite(args.learning_rate) or not 0 < args.learning_rate <= .01:
        parser.error("wall budget (0,1800] and learning rate (0,.01] required")
    if args.lr_final is not None and (not math.isfinite(args.lr_final) or not 0 < args.lr_final <= args.learning_rate):
        parser.error("schedule end rate must be finite, positive and no greater than the learning rate")
    if args.lr_final is not None and args.type_head_rate_factor != 1:
        parser.error("the learning-rate schedule is declared on the published single-group optimiser")
    if not 1 <= args.evaluate_every <= 1024 or not 3 <= args.development_groups <= 12:
        parser.error("evaluation interval 1..1024 and development groups 3..12 required")
    if not 1 <= args.checkpoint_every <= 1024 or args.max_new_steps is not None and not 1 <= args.max_new_steps <= 4096:
        parser.error("checkpoint interval1..1024 and new-update cap1..4096 required")
    if args.profile_square and (args.steps != 2 or args.admission is not None or args.resume_from is not None or args.max_new_steps is not None):
        parser.error("square resource profile requires two optimizer steps without prior admission")
    if not args.profile_square and args.admission is None:
        parser.error("a matching warm-optimizer square resource admission is required")
    if args.box_extent_weight and (args.lattice_support or args.geometry_final_only):
        parser.error("the box-extent lever is declared on the published box head and geometry stage")
    if args.support_anchor_weight and (args.lattice_support or args.geometry_final_only or
                                       args.box_extent_weight):
        parser.error("the capacity lever is declared on the published box head, stage policy and objective")
    if args.lattice_residual and (args.lattice_support or args.geometry_final_only or
                                  args.box_extent_weight or args.support_anchor_weight):
        parser.error("the lattice-residual lever is declared on the published box head, objective "
                     "and geometry stage policy")
    if args.occupancy_lattice_free == 1 and args.occupancy_stack_monotone:
        # The single free channel replaces the lattice-sampled read, so it has nothing to be an envelope of;
        # the summed form (2) is the one that may carry the envelope, and it is measurable with or without it.
        parser.error("the single lattice-free channel replaces the lattice-sampled read, so it is not declared "
                     "beside --occupancy-stack-monotone 1; the summed form is --occupancy-lattice-free 2")
    if args.occupancy_free_stencil and args.occupancy_lattice_free != 2:
        parser.error("the local stencil is detail for the free channel, so it is declared only beside "
                     "--occupancy-lattice-free 2")
    if args.occupancy_pair_gap and args.occupancy_lattice_free != 2:
        parser.error("the pair-margin term supervises the free channel's own read, so it is declared only beside "
                     "--occupancy-lattice-free 2")
    if args.occupancy_pair_gap and args.occupancy_pair_rank:
        parser.error("the symmetric pair margin and the one-sided pair order are two statements about the same pairs, "
                     "so an arm declares one or the other")
    if args.occupancy_pair_gap and (args.occupancy_midpoint or args.occupancy_anchor or
                                    args.occupancy_quantile_anchor or args.occupancy_dense_hinge or
                                    args.occupancy_margin_hinge or args.occupancy_centre):
        parser.error("the pair-margin term is invariant to a board's level, so the five boundary/margin terms and "
                     "structural centring are refused beside it; the envelope and the crossing rule compose with it "
                     "by declaration")
    if args.occupancy_pair_rank and args.occupancy_lattice_free != 2:
        parser.error("the pair-order term supervises the free channel's own read, so it is declared only beside "
                     "--occupancy-lattice-free 2")
    if args.occupancy_pair_rank and (args.occupancy_midpoint or args.occupancy_anchor or
                                     args.occupancy_quantile_anchor or args.occupancy_dense_hinge or
                                     args.occupancy_margin_hinge or args.occupancy_centre):
        # The pair term is a statement about the read's ordering; the five boundary/margin terms and structural centring
        # are statements about where the read sits. The pair term is invariant to a board's common shift, so a level term
        # beside it would be supervised on a quantity this term does not constrain, and an arm carrying both could not
        # attribute a change to either. The structural transforms (envelope, crossing rule) compose by declaration: they
        # decide at the read's own split rather than moving it, which is exactly the composition M25 asks about.
        parser.error("the pair-order term supervises the read's ordering and is invariant to its level, so the five "
                     "boundary/margin terms and structural centring are refused beside it; the envelope and the "
                     "crossing rule compose with it by declaration")
    if args.occupancy_lattice_free and (args.lattice_residual or args.lattice_support or
                                       args.geometry_final_only or args.box_extent_weight or
                                       args.support_anchor_weight):
        parser.error("the lattice-free readout lever is declared on the published model")
    if args.lattice_board_refine and (args.lattice_residual or args.lattice_support or
                                      args.geometry_final_only or args.box_extent_weight or
                                      args.support_anchor_weight or args.occupancy_lattice_free):
        parser.error("the board-refinement lever is declared on the published decode and model")
    if args.accumulation_policy is not None:
        size = ACCUMULATION_POLICIES[args.accumulation_policy][0]
        expected = "all-class-fit-v1" if args.accumulation_policy == "ordered-eight-v1" else "all-class-interleaved-v1"
        if (args.model != "pilot" or args.component_capacity or args.layer_evidence or
                args.learning_rate != .001 or args.seed != 20261026):
            parser.error("accumulated fit requires pilot M2, no optional heads, seed 20261026 and learning rate .001")
        if args.curriculum != expected:
            parser.error("accumulation policy requires its predeclared "+expected+" curriculum")
        if args.resume_from is not None and args.resume_from.resolve() == args.output.resolve():
            parser.error("an accumulated segment needs its own output directory")
        if not args.profile_square and args.steps != update_count(args.accumulation_policy):
            parser.error("an accumulated arm consumes exactly "+str(update_count(args.accumulation_policy))+" optimizer updates")
        if args.checkpoint_every % size or args.evaluate_every % size:
            parser.error("checkpoint and evaluation intervals must be complete-group boundaries")
    return args


def compatible_config(source, current, allow_absent=()):
    """A source checkpoint may differ only by declared keys it predates, never in a value."""
    if type(source) is not dict or type(current) is not dict:
        return False
    if set(source)-set(current):
        return False
    extra = set(current)-set(source)
    if extra and not extra <= set(allow_absent):
        return False
    return all(source[key] == current[key] for key in source)


def staged_mode(summary, *, freeze_backbone, init_weights_from):
    """Resource admission must pin the staged mode, never inherit it by default."""
    arguments = summary.get("arguments", {})
    declared = arguments.get("freeze_backbone", False)
    source = arguments.get("init_weights_from")
    if type(declared) is not bool or declared is not freeze_backbone:
        return False
    if (source is None) is not (init_weights_from is None):
        return False
    return source is None or str(source) == str(init_weights_from)


# Declared prefix list of tensors a continuation may add over the published checkpoint. It is one
# constant rather than a literal inside the check so a new declared module cannot be refused for
# matching its parent's name only up to the first underscore (V-104).
INITIAL_SOURCE_ADDED_PREFIXES = ("occupancy_neighborhood.", "occupancy_pixel_query.",
                                 "pixel_query_level", "read_calibration",
                                 "occupancy_detail_tap.",
                                 "occupancy_resolution_branch.", "occupancy_deform.",
                                 "occupancy_deform_coarse.", "support_box.", "lattice_residual.",
                                 "occupancy_lattice_free.", "occupancy_lattice_gate",
                                 # M22: the stencil head is declared by the arm that carries it, and
                                 # an arm without it is the same function, so a source checkpoint
                                 # that predates the field is a legitimate start.
                                 "occupancy_free_stencil.",
                                 "lattice_board_refine.", "occupancy_column.")


def complete_recovery_contract(contract, config):
    """Every declared occupancy lever the config carries must reach the contract.

    The assignments above are hand-maintained, and V-147 found the failure mode: a field can be added to the model and
    the parser, be recorded in the arguments, and still be absent from the contract, which the admission join reads - so
    the arm that declares it is refused by its own admission. Filling from the config makes the contract complete by
    construction rather than by remembering.
    """
    for name in sorted(config):
        if name.startswith("occupancy_") and name not in contract:
            contract[name] = config[name]
    return contract


def owned_initial_source(source, *, output):
    """A staged start must be a different saved run owned beside this one.

    The driver re-executes from the verified archive under <runs>/<name>/code, so
    its ROOT there is the archive copy and cannot name the project's runs. The run
    being written is the only reliable anchor for the owned runs directory.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    if source.parent != output.parent or source == output:
        raise ValueError("initial weights must come from a different owned run")
    return source


def validate_admission(summary, job, *, catalogue, model, config, ceiling, development_groups, supervisor_sha, curriculum="stream-v1", type_head_rate_factor=1, identity_factor=1, endpoint_term="joint", occupancy_factor=1, positive_weight=1, accumulation_policy=None, profile_work_items=2, freeze_backbone=False, init_weights_from=None, max_side=512, occupancy_context=0, occupancy_pixel_query=0, pixel_query_level=0, occupancy_read_standardise=0, occupancy_read_level=0, occupancy_balance=0, occupancy_mass=0, occupancy_detail=0, occupancy_detail_stencil=0, occupancy_resolution=0, occupancy_deform=0, geometry_final_only=0, lattice_support=0, box_extent_weight=0.0, support_anchor_weight=0.0,
                   lattice_residual=0, occupancy_lattice_free=0, occupancy_free_stencil=0,
                   lattice_board_refine=0, occupancy_pair_rank=0, occupancy_pair_gap=0,
                   occupancy_placed=0, occupancy_column=0, occupancy_offset_rows=0,
                   occupancy_normalise=0, occupancy_margin_hinge=0, occupancy_anchor=0,
                   occupancy_quantile_anchor=0, occupancy_dense_hinge=0, occupancy_midpoint=0,
                   occupancy_centre=0, occupancy_stack_monotone=0, occupancy_crossing_threshold=0,
                   occupancy_skyline_recovery=0, occupancy_tap_span=0):
    if (type(curriculum) is not str or curriculum not in ("stream-v1", "all-class-fit-v1", "all-class-interleaved-v1") or
            summary.get("curriculum", "stream-v1") != curriculum):
        raise ValueError("curriculum differs from resource admission")
    if not staged_mode(summary, freeze_backbone=freeze_backbone, init_weights_from=init_weights_from):
        raise ValueError("staged refinement mode differs from resource admission")
    if "curriculum" in summary and (summary.get("arguments", {}).get("curriculum") != curriculum or
                                    summary.get("recovery_contract", {}).get("curriculum") != curriculum):
        raise ValueError("resource arguments/recovery curriculum identity differs")
    if (type(type_head_rate_factor) is not int or type_head_rate_factor not in (1, 4) or
            summary.get("arguments", {}).get("type_head_rate_factor", 1) != type_head_rate_factor or
            summary.get("recovery_contract", {}).get("type_head_rate_factor", 1) != type_head_rate_factor):
        raise ValueError("type-head rate factor differs from resource admission")
    if (type(identity_factor) is not int or identity_factor not in (1, 4) or
            summary.get("arguments", {}).get("component_identity_factor", 1) != identity_factor or
            summary.get("recovery_contract", {}).get("component_identity_factor", 1) != identity_factor):
        raise ValueError("component identity factor differs from resource admission")
    from tetris_vision.components import require_endpoint_term
    from tetris_vision.losses import (require_box_extent_weight, require_geometry_final_only,
                                     require_lattice_support, require_lattice_residual,
                                     require_occupancy_lattice_free, require_occupancy_free_stencil,
                                     require_support_anchor_weight,
                                     require_lattice_board_refine, require_occupancy_pair_rank,
                                     require_occupancy_pair_gap,
                                     require_occupancy_context, require_occupancy_pixel_query,
                                     require_pixel_query_level, require_occupancy_read_standardise,
                                     require_occupancy_balance, require_occupancy_read_level,
                                     require_occupancy_mass,
                                     require_occupancy_detail, require_occupancy_detail_stencil,
                                     require_occupancy_resolution, require_occupancy_factor,
                                     require_occupancy_deform, require_positive_weight,
                                     require_occupancy_placed, require_occupancy_column,
                                     require_occupancy_offset_rows, require_occupancy_normalise,
                                     require_occupancy_margin_hinge, require_occupancy_anchor,
                                     require_occupancy_quantile_anchor, require_occupancy_dense_hinge,
                                     require_occupancy_midpoint, require_occupancy_centre,
                                     require_occupancy_stack_monotone,
                                     require_occupancy_crossing_threshold,
                                     require_occupancy_skyline_recovery, require_occupancy_tap_span)
    require_occupancy_free_stencil(occupancy_free_stencil, summary)
    require_occupancy_pair_rank(occupancy_pair_rank, summary)
    require_occupancy_pair_gap(occupancy_pair_gap, summary)
    require_geometry_final_only(geometry_final_only, summary)
    require_lattice_support(lattice_support, summary)
    require_endpoint_term(endpoint_term, summary)
    require_occupancy_factor(occupancy_factor, summary)
    require_positive_weight(positive_weight, summary)
    require_occupancy_context(occupancy_context, summary)
    require_occupancy_pixel_query(occupancy_pixel_query, summary)
    require_pixel_query_level(pixel_query_level, summary)
    require_occupancy_read_standardise(occupancy_read_standardise, summary)
    require_occupancy_read_level(occupancy_read_level, summary)
    require_occupancy_mass(occupancy_mass, summary)
    require_occupancy_balance(occupancy_balance, summary)
    require_occupancy_detail(occupancy_detail, summary)
    require_occupancy_detail_stencil(occupancy_detail_stencil, summary)
    require_box_extent_weight(box_extent_weight, summary)
    require_support_anchor_weight(support_anchor_weight, summary)
    require_lattice_residual(lattice_residual, summary)
    require_lattice_board_refine(lattice_board_refine, summary)
    require_occupancy_lattice_free(occupancy_lattice_free, summary)
    require_occupancy_resolution(occupancy_resolution, summary)
    require_occupancy_deform(occupancy_deform, summary)
    require_occupancy_placed(occupancy_placed, summary)
    require_occupancy_column(occupancy_column, summary)
    require_occupancy_offset_rows(occupancy_offset_rows, summary)
    require_occupancy_normalise(occupancy_normalise, summary)
    require_occupancy_margin_hinge(occupancy_margin_hinge, summary)
    require_occupancy_anchor(occupancy_anchor, summary)
    require_occupancy_quantile_anchor(occupancy_quantile_anchor, summary)
    require_occupancy_dense_hinge(occupancy_dense_hinge, summary)
    require_occupancy_midpoint(occupancy_midpoint, summary)
    require_occupancy_centre(occupancy_centre, summary)
    require_occupancy_stack_monotone(occupancy_stack_monotone, summary)
    require_occupancy_crossing_threshold(occupancy_crossing_threshold, summary)
    require_occupancy_skyline_recovery(occupancy_skyline_recovery, summary)
    require_occupancy_tap_span(occupancy_tap_span, summary)
    warm = summary.get("warm_optimizer_backward", {})
    if accumulation_policy is not None and (
            summary.get("accumulation_policy") != accumulation_policy or
            summary.get("images_per_update") != ACCUMULATION_POLICIES[accumulation_policy][0] or
            summary.get("arguments", {}).get("accumulation_policy") != accumulation_policy or
            warm.get("policy") != accumulation_policy or
            warm.get("group_size") != ACCUMULATION_POLICIES[accumulation_policy][0] or
            warm.get("microsteps") != 2*ACCUMULATION_POLICIES[accumulation_policy][0] or
            warm.get("recovery_exact") is not True or
            not summary.get("optimizer_updates_per_second", 0) > 0):
        raise ValueError("accumulation policy/group-size profile differs from resource admission")
    if (summary.get("status") != "complete" or summary.get("steps") != 2 or not summary.get("profile_square") or
            summary.get("resource_profile_version") != 3 or not summary.get("training_recovery_profile_exact") or
            warm.get("step") != 2 or warm.get("input_pixels") != 512**2 or
            not warm.get("all_parameters_initialized") or not warm.get("backward_completed") or warm.get("state_bytes", 0) <= 0 or
            not summary.get("checkpoint_round_trip_exact") or
            type(summary.get("max_input_pixels_seen")) is not int or
            not 512**2 <= summary["max_input_pixels_seen"] <= max_side**2 or
            summary.get("native_cache_limits") != NATIVE_CACHE_LIMITS or
            any(summary.get(key) != value for key, value in {"max_side": max_side, "threads": 1, "workers": 0, "prefetch": 0, "tensor_cache_bytes": 0}.items()) or
            job.get("status") != "passed" or job.get("exit_code") != 0 or job.get("work_items") != profile_work_items):
        raise ValueError("complete two-step square warm-optimizer resource admission required")
    if (summary["catalogue_sha256"] != catalogue or summary["model_kind"] != model or summary["config"] != config or
            summary["job_ceiling_bytes"] != ceiling or job["memory_bytes"] != ceiling or
            summary["development_groups"] < development_groups or not summary["cpu_only"] or
            summary.get("system_pressure_guard") != "GetPerformanceInfo/0.2s" or job.get("system_pressure_guard") != "GetPerformanceInfo/0.2s" or
            summary.get("supervisor_script_sha256") != supervisor_sha or job.get("supervisor_script_sha256") != supervisor_sha):
        raise ValueError("source/config/input/CPU/memory envelope differs from resource admission")


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    replace_retry(temporary, path)


def equal_tree(left, right):
    """Exact nested tensor parity for warm recovery and M2/M3 final outputs."""
    import torch
    if torch.is_tensor(left):
        return (torch.is_tensor(right) and left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right))
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(equal_tree(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(equal_tree(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def cosine_rate(start, final, step, total):
    """Cosine-decayed learning rate for one step of a `total`-step segment.

    Declared by `docs/M3_OPTIMISER_SCHEDULE.md`: step 1 is exactly `start`, step `total`
    is exactly `final`, and the curve is non-increasing between them. The endpoints are
    returned literally rather than evaluated, so neither end of the schedule depends on
    floating-point cancellation in the cosine.
    """
    if not math.isfinite(start) or not math.isfinite(final) or not 0 < final <= start:
        raise ValueError("schedule requires finite 0 < final <= start")
    if type(step) is not int or type(total) is not int or total < 1 or not 1 <= step <= total:
        raise ValueError("schedule step must lie inside 1..total")
    if step == 1:
        return float(start)
    if step == total:
        return float(final)
    progress = (step-1)/(total-1)
    return float(final+(start-final)*.5*(1+math.cos(math.pi*progress)))


def make_optimizer(model, learning_rate, factor):
    """Keep the original single group by default; isolate exactly the type head otherwise."""
    import torch
    if factor == 1:
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=.01)
    if factor != 4:
        raise ValueError("unsupported type-head rate factor")
    named = dict(model.named_parameters())
    head_names = {"layer_type.weight", "layer_type.bias"}
    if not head_names <= named.keys():
        raise ValueError("expected shared type head missing")
    rest = [parameter for name, parameter in named.items() if name not in head_names]
    head = [named["layer_type.weight"], named["layer_type.bias"]]
    return torch.optim.AdamW([{"params": rest}, {"params": head, "lr": learning_rate * factor}],
                             lr=learning_rate, weight_decay=.01)


def capture_initial_tensors(model, path):
    """Save the actual initialized CPU state before evaluation or optimization."""
    import torch
    with path.open("xb") as output:
        torch.save(model.state_dict(), output)
        output.flush()
        os.fsync(output.fileno())
    saved = torch.load(path, map_location="cpu", weights_only=True)
    current = model.state_dict()
    if saved.keys() != current.keys() or any(
            saved[name].dtype != value.dtype or saved[name].shape != value.shape or
            not torch.equal(saved[name].contiguous().reshape(-1).view(torch.uint8),
                            value.contiguous().reshape(-1).view(torch.uint8))
            for name, value in current.items()):
        raise ValueError("initial state serialization changed tensor bytes")


def accumulation_profile(*, run, model, optimizer, contract, policy, completed, maximum_pixels,
                         clock=time.monotonic):
    """Measure warm Adam state and an actual boundary reload for one group size."""
    import copy
    import torch
    from probe_model import bounded_job_snapshot
    from training_snapshot import load_training_snapshot
    if type(completed) is not int or completed < 1:
        raise ValueError("a completed accumulation boundary is required")
    parameters = [p for p in model.parameters() if p.requires_grad]
    initialized = all(
        p in optimizer.state and all(key in optimizer.state[p] and optimizer.state[p][key].shape == p.shape
                                     for key in ("exp_avg", "exp_avg_sq"))
        and float(optimizer.state[p]["step"]) == completed for p in parameters)
    if not parameters or not initialized:
        raise RuntimeError("warm accumulation profile requires resident Adam state")
    state_bytes = sum(value.numel()*value.element_size() for state in optimizer.state.values()
                      for value in state.values() if torch.is_tensor(value))
    warm = bounded_job_snapshot()
    reload_started = clock()
    expected = copy.deepcopy((model.state_dict(), optimizer.state_dict(), torch.get_rng_state()))
    recovery = load_training_snapshot(run, model, optimizer, contract=contract)
    exact = recovery["step"] == completed and equal_tree(
        expected, (model.state_dict(), optimizer.state_dict(), torch.get_rng_state()))
    reload_seconds = clock()-reload_started
    del expected, recovery
    reload = bounded_job_snapshot()
    size = ACCUMULATION_POLICIES[policy][0]
    return {"step": completed, "policy": policy, "group_size": size,
            "images": completed*size, "microsteps": completed*size,
            "input_pixels": maximum_pixels, "parameter_tensors": len(parameters),
            "all_parameters_initialized": True, "state_bytes": state_bytes,
            "backward_completed": True, "warm_peak_commit_bytes": warm["peak_job_commit_bytes"],
            "reload_peak_commit_bytes": reload["peak_job_commit_bytes"],
            "reload_commit_delta_bytes": reload["peak_job_commit_bytes"]-warm["peak_job_commit_bytes"],
            "reload_seconds": reload_seconds, "recovery_exact": exact,
            "device": "cpu", "device_memory_bytes": 0}


def run_accumulation_loop(*, run, policy, contract, model, optimizer, updates_total, recipe_for, loss_for,
                          checkpoint_every, evaluate_every, wall_seconds, started, on_group,
                          on_evaluation=None, on_progress=None, start_update=1, clock=time.monotonic):
    """Execute whole groups through the durable journal; stop only at a committed boundary.

    A caller supplies the actual recipe and loss for each image cursor. Every
    group is validated, optimized, journaled and only then snapshotted, so a wall
    or process stop can never publish a partial group.
    """
    from accumulation_checkpoint import append_group, save_boundary
    from accumulation_journal import execute_group
    size = ACCUMULATION_POLICIES[policy][0]
    if (type(updates_total) is not int or not 1 <= updates_total <= update_count(policy) or
            type(start_update) is not int or not 1 <= start_update <= updates_total or
            type(checkpoint_every) is not int or not 1 <= checkpoint_every or checkpoint_every % size or
            type(evaluate_every) is not int or not 1 <= evaluate_every or evaluate_every % size):
        raise ValueError("bounded complete-group accumulation cadence required")
    state = pointer = None
    snapshotted = 0
    for update in range(start_update, updates_total+1):
        if clock()-started >= wall_seconds:
            break
        recipes = [recipe_for(image, index, view) for image, index, view in group_specs(policy, update)]
        entry = execute_group(model, optimizer, policy, update, recipes, loss_for)
        state = append_group(run, policy, entry)
        # A boundary snapshot must never see stepped gradients as pending work.
        optimizer.zero_grad(set_to_none=True)
        on_group(update, state, entry)
        images = state["images_seen"]
        if images % evaluate_every == 0 and on_evaluation is not None:
            on_evaluation(update, state)
        if images % checkpoint_every == 0 or update == updates_total or state["status"] == "complete":
            pointer = save_boundary(run, model, optimizer, contract)
            snapshotted = images
        if on_progress is not None:
            on_progress(update, state, entry["mean_loss"])
    if state is not None and snapshotted != state["images_seen"]:
        pointer = save_boundary(run, model, optimizer, contract)
        snapshotted = state["images_seen"]
    return {"state": state, "pointer": pointer, "snapshotted_images": snapshotted}


def fit_correct(semantic, target):
    """Per-cell semantic correctness over the labelled board, sized by the target."""
    from board_window import visible_board_for_target
    return visible_board_for_target(semantic, target, "semantic").argmax(-1) == target["semantic"]


def main(argv=None, *, _archive=None, _started=None):
    args = arguments(argv)
    ceiling = require_bounded_job()
    args.output = args.output.resolve()
    if _archive is None:
        args.output.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        atomic_json(args.output/"progress.json", {"status": "capturing_source", "pid": os.getpid(), "step": 0})
        code = args.output/"code"
        archive = capture_source(ROOT, code)
        # Execute the driver itself from the verified archive, so tracebacks
        # cannot read later parent edits. This stays in the same native Job/PID.
        sys.path[:0] = [str(code/"scripts"), str(code)]
        sys.dont_write_bytecode = True
        for name in ("probe_model", "source_snapshot", "atomic_write", "run_bounded",
                     "accumulation_contract", "accumulation_journal", "accumulation_checkpoint",
                     "training_snapshot", "m2_interleaved_order"):
            sys.modules.pop(name, None)
        spec = importlib.util.spec_from_file_location("frozen_training_driver", code/"scripts/train_m2_stream.py")
        driver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(driver)
        return driver.main(argv, _archive=archive, _started=started)
    started = _started
    progress = {"status": "capturing_source", "pid": os.getpid(), "step": 0, "requested_steps": args.steps}
    atomic_json(args.output/"progress.json", progress)
    code = args.output/"code"
    archive = verify_source(code, _archive["catalogue_sha256"])
    if not Path(__file__).resolve().is_relative_to(code):
        raise RuntimeError("trainer main did not execute from archived source")
    supervisor_sha = archive["files"]["scripts/run_bounded.py"]
    if os.environ.get("TETRIS_VISION_SYSTEM_GUARD") != "GetPerformanceInfo/0.2s" or os.environ.get("TETRIS_VISION_SUPERVISOR_SHA256") != supervisor_sha:
        raise RuntimeError("supervisor source/OS pressure guard does not match the archived runner")
    sys.path[:0] = [str(code/"scripts"), str(code)]
    sys.dont_write_bytecode = True
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ.update(NATIVE_CACHE_LIMITS)
    import torch
    from dataclasses import asdict, replace
    from m2_curriculum import FIT_POLICIES, distinct_training_groups, probe_specs, replay_case, sample_for, training_case
    from m2_fit_contract import summarize_fit
    from training_snapshot import load_training_snapshot, save_training_snapshot
    from tetris_vision.checkpoint import file_digest, load_checkpoint, save_checkpoint
    from tetris_vision.dataset import source_digest
    from tetris_vision.losses import PIECE_NAMES, diagnostic_metrics, joint_loss, targets_for
    from tetris_vision.model import ModelConfig, TetrisStateNet, architecture_name
    from tetris_vision.preprocess import prepare_image
    import tetris_vision.model as model_module
    if (not Path(model_module.__file__).resolve().is_relative_to(code) or
            any(not Path(function.__code__.co_filename).resolve().is_relative_to(code) for function in (sample_for, training_case, summarize_fit))):
        raise RuntimeError("runtime did not import from the verified source archive")
    if args.curriculum == "all-class-interleaved-v1":
        from m2_interleaved_order import distinct_groups, interleaved_spec
        if any(not Path(function.__code__.co_filename).resolve().is_relative_to(code)
               for function in (distinct_groups, interleaved_spec)):
            raise RuntimeError("interleaved order implementation is not frozen")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    config = ModelConfig.pilot() if args.model == "pilot" else ModelConfig()
    if args.occupancy_balance:
        # The declared data-derived budget is passed through the existing class-budget
        # parameter, so the level comes from the loss and no new loss path is introduced.
        args.occupancy_positive_weight = -1
    config = replace(config, component_capacity=args.component_capacity, layer_evidence=args.layer_evidence,
                     component_identity_readout=args.component_identity_readout,
                     component_endpoint_term=args.component_endpoint_term,
                     occupancy_context=args.occupancy_context,
                     occupancy_pixel_query=args.occupancy_pixel_query,
                     pixel_query_level=args.pixel_query_level,
                     occupancy_read_standardise=args.occupancy_read_standardise,
                     occupancy_read_level=args.occupancy_read_level,
                     occupancy_balance=args.occupancy_balance,
                     occupancy_detail=args.occupancy_detail,
                     occupancy_detail_stencil=args.occupancy_detail_stencil,
                     occupancy_resolution=args.occupancy_resolution,
                     occupancy_deform=args.occupancy_deform,
                     occupancy_column=args.occupancy_column,
                     occupancy_offset_rows=args.occupancy_offset_rows,
                     occupancy_normalise=args.occupancy_normalise,
                     occupancy_centre=args.occupancy_centre,
                     occupancy_stack_monotone=args.occupancy_stack_monotone,
                     occupancy_crossing_threshold=args.occupancy_crossing_threshold,
                     occupancy_skyline_recovery=args.occupancy_skyline_recovery,
                     occupancy_tap_span=args.occupancy_tap_span,
                     lattice_support=args.lattice_support, lattice_residual=args.lattice_residual,
                     occupancy_lattice_free=args.occupancy_lattice_free,
                     occupancy_free_stencil=args.occupancy_free_stencil,
                     lattice_board_refine=args.lattice_board_refine,
                     dimension=args.trunk_dimension)
    config_json = json.loads(json.dumps(asdict(config)))
    accumulation = None
    if args.accumulation_policy is not None:
        accumulation = make_contract(args.accumulation_policy,
                                     source_catalogue_sha256=archive["catalogue_sha256"], config=config_json)
    if args.admission is not None:
        admit = args.admission.resolve()
        validate_admission(json.loads((admit/"summary.json").read_text()),
                           json.loads((admit.parent/(admit.name+"-job.json")).read_text()),
                           catalogue=archive["catalogue_sha256"], model=args.model, config=config_json,
                           ceiling=ceiling, development_groups=args.development_groups, supervisor_sha=supervisor_sha, curriculum=args.curriculum,
                           type_head_rate_factor=args.type_head_rate_factor, identity_factor=args.component_identity_factor,
                           endpoint_term=args.component_endpoint_term, occupancy_factor=args.occupancy_factor,
                           positive_weight=args.occupancy_positive_weight, box_extent_weight=args.box_extent_weight,
                           support_anchor_weight=args.support_anchor_weight,
                           occupancy_context=args.occupancy_context,
                           occupancy_pixel_query=args.occupancy_pixel_query,
                           pixel_query_level=args.pixel_query_level,
                           occupancy_read_standardise=args.occupancy_read_standardise,
                           occupancy_read_level=args.occupancy_read_level,
                           occupancy_balance=args.occupancy_balance,
                           occupancy_mass=args.occupancy_mass,
                           occupancy_detail=args.occupancy_detail,
                           occupancy_detail_stencil=args.occupancy_detail_stencil,
                           occupancy_resolution=args.occupancy_resolution,
                           occupancy_deform=args.occupancy_deform,
                           occupancy_placed=args.occupancy_placed,
                           occupancy_margin_hinge=args.occupancy_margin_hinge,
                           occupancy_anchor=args.occupancy_anchor,
                           occupancy_quantile_anchor=args.occupancy_quantile_anchor,
                           occupancy_dense_hinge=args.occupancy_dense_hinge,
                           occupancy_midpoint=args.occupancy_midpoint,
                           occupancy_centre=args.occupancy_centre,
                     occupancy_stack_monotone=args.occupancy_stack_monotone,
                     occupancy_crossing_threshold=args.occupancy_crossing_threshold,
                     occupancy_skyline_recovery=args.occupancy_skyline_recovery,
                     occupancy_tap_span=args.occupancy_tap_span,
                           occupancy_column=args.occupancy_column,
                           occupancy_offset_rows=args.occupancy_offset_rows,
                           occupancy_normalise=args.occupancy_normalise,
                           geometry_final_only=args.geometry_final_only,
                           lattice_support=args.lattice_support,
                           lattice_residual=args.lattice_residual,
                           occupancy_lattice_free=args.occupancy_lattice_free,
                           occupancy_free_stencil=args.occupancy_free_stencil,
                           occupancy_pair_rank=args.occupancy_pair_rank,
                           occupancy_pair_gap=args.occupancy_pair_gap,
                           lattice_board_refine=args.lattice_board_refine,
                           accumulation_policy=args.accumulation_policy,
                           freeze_backbone=args.freeze_backbone, init_weights_from=args.init_weights_from,
                           max_side=args.capture_max_side,
                           profile_work_items=2*(ACCUMULATION_POLICIES[args.accumulation_policy][0]
                                                 if args.accumulation_policy else 1))
    model = TetrisStateNet(config)
    initial_origin = None
    if args.init_weights_from is not None:
        source = owned_initial_source(args.init_weights_from, output=args.output)
        source_summary = json.loads((source/"summary.json").read_text())
        source_model, source_manifest = load_checkpoint(source/"checkpoint")
        if (source_manifest["architecture"] != architecture_name(config) or
                not compatible_config(source_manifest["config"], config_json,
                                      allow_absent=(                                                    "occupancy_context", "occupancy_pixel_query",
                                                    "occupancy_detail", "pixel_query_level",
                                                    "occupancy_read_standardise",
                                                    "occupancy_balance", "occupancy_read_level",
                                                    "occupancy_mass",
                                                    "occupancy_detail_stencil",
                                                    "occupancy_resolution", "occupancy_deform",
                                                    "lattice_support",
                                                    "lattice_residual",
                                                    "occupancy_lattice_free",
                                                    # M22: default 0 constructs no stencil head, so a
                                                    # source checkpoint that predates the field is the
                                                    # same function and a legitimate initial state.
                                                    "occupancy_free_stencil",
                                                    "lattice_board_refine",
                                                    "occupancy_column",
                                                    "occupancy_offset_rows",
                                                    "occupancy_normalise",
                                                    # M14: a default-0 structural option adds no parameter, so a
                                                    # source checkpoint that predates the field is the same
                                                    # function and a legitimate initial state for either arm.
                                                    "occupancy_centre",
                                                    "occupancy_stack_monotone",
                                                    "occupancy_crossing_threshold",
                                                    "occupancy_skyline_recovery", "occupancy_tap_span")) or
                source_summary["checkpoint_sha256"] != source_manifest["weights_sha256"] or
                source_summary.get("curriculum") != args.curriculum or
                source_summary["arguments"].get("seed") != args.seed or
                float(source_summary["arguments"].get("learning_rate", 0)) != args.learning_rate):
            raise ValueError("initial-weight source architecture/config/curriculum/seed differs")
        source_state = source_model.state_dict()
        missing = sorted(set(model.state_dict())-set(source_state))
        if set(source_state)-set(model.state_dict()) or any(
                not key.startswith(INITIAL_SOURCE_ADDED_PREFIXES)
                for key in missing):
            raise ValueError("initial-weight source tensors differ from the declared representation")
        model.load_state_dict(source_state, strict=False)
        if any(not torch.equal(value, model.state_dict()[name]) for name, value in source_state.items()):
            raise ValueError("initial-weight tensors changed while loading")
        initial_origin = {"run": str(source), "catalogue_sha256": source_summary["catalogue_sha256"],
                          "weights_sha256": source_summary["checkpoint_sha256"],
                          "steps": source_summary["steps"], "curriculum": source_summary["curriculum"],
                          "seed": source_summary["arguments"]["seed"]}
    if args.freeze_backbone:
        head_names = {"layer_type.weight", "layer_type.bias"}
        named = dict(model.named_parameters())
        if not head_names <= named.keys():
            raise ValueError("expected shared type head missing")
        for name, parameter in named.items():
            parameter.requires_grad = name in head_names
        if any(parameter.requires_grad for name, parameter in model.named_parameters() if name not in head_names):
            raise RuntimeError("frozen-backbone stage left another parameter trainable")
    initial_path = args.output / "initial-weights.pt"
    capture_initial_tensors(model, initial_path)
    initial_sha256 = file_digest(initial_path)
    optimizer = make_optimizer(model, args.learning_rate, args.type_head_rate_factor)
    scheduled_base_rates = tuple(group["lr"] for group in optimizer.param_groups)
    maximum_pixels, evaluation_identities = 0, {}
    warm_optimizer_backward = None
    recovery_profile_exact = False

    def sample(group, view, split, *, profile_square=False):
        return sample_for(group, view, split, args.seed, profile_square=profile_square,
                          curriculum=args.curriculum, component_capacity=args.component_capacity,
                          max_side=args.capture_max_side)

    def prepare(case):
        nonlocal maximum_pixels
        prepared = prepare_image(case.frame.image, max_side=args.capture_max_side)
        maximum_pixels = max(maximum_pixels, prepared.pixels.shape[-2]*prepared.pixels.shape[-1])
        if maximum_pixels > args.capture_max_side**2:
            raise ValueError(f"{args.capture_max_side}-square input budget exceeded")
        target = targets_for(case.frame.state, prepared, component_capacity=args.component_capacity,
                             layer_evidence=args.layer_evidence)
        if int(target["known"].sum()) != 200:
            raise ValueError("curriculum input admission requires all 200 visible locked cells")
        return prepared, target

    @torch.no_grad()
    def evaluate(split, groups, persist=False):
        model.eval()
        records, per_case = [], []
        for group, view in probe_specs(split, groups, args.curriculum):
            case = sample(group, view, split)
            prepared, target = prepare(case)
            output = model(prepared.pixels[None], prepared.valid[None])
            loss, _ = joint_loss(output, target, identity_factor=args.component_identity_factor, endpoint_term=args.component_endpoint_term, occupancy_factor=args.occupancy_factor, positive_weight=args.occupancy_positive_weight, geometry_final_only=args.geometry_final_only, box_extent_weight=args.box_extent_weight, support_anchor_weight=args.support_anchor_weight, occupancy_placed=args.occupancy_placed, occupancy_margin_hinge=args.occupancy_margin_hinge, occupancy_anchor=args.occupancy_anchor, occupancy_quantile_anchor=args.occupancy_quantile_anchor, occupancy_dense_hinge=args.occupancy_dense_hinge, occupancy_midpoint=args.occupancy_midpoint, occupancy_pair_rank=args.occupancy_pair_rank, occupancy_pair_gap=args.occupancy_pair_gap,
occupancy_mass=args.occupancy_mass)
            layers_right = layers_known = 0
            fit_roles = {}
            for layer, role in enumerate(("active", "ghost")):
                known = bool(target["layer_known"][0, layer])
                observed = bool(target["pose_known"][0, layer])
                right = None
                if known:
                    layers_known += 1
                    final = output["final"]
                    right = int(final["layer_type"][0, layer].argmax()) == int(target["layer_type"][0, layer]) and bool(final["layer_known"][0, layer] > 0)
                    if observed:
                        right = (right and int(final["pose_column"][0, layer].argmax()) == int(target["column"][0, layer]) and
                                 int(final["pose_row"][0, layer].argmax()) == int(target["row"][0, layer]) and
                                 bool(target["rotations"][0, layer, final["pose_rotation"][0, layer].argmax()]))
                    layers_right += int(right)
                if args.curriculum in FIT_POLICIES and split == "training":
                    kind = int(target["layer_type"][0, layer])
                    fit_roles[role] = {"known": known, "observed": observed, "exact": right,
                                       "class": (PIECE_NAMES[kind-1] if kind else "absent") if known else "unknown"}
            per_case.append({"sample_id": case.record["sample_id"], "loss": float(loss),
                             "exact_known_layers": layers_right, "known_layers": layers_known,
                             **diagnostic_metrics(output, target)})
            if args.curriculum in FIT_POLICIES and split == "training":
                typed = target["typed"]
                correct = fit_correct(output["final"]["semantic"], target) & typed
                per_case[-1].update(group_id=case.record["group_id"],
                                    fit_case={"semantic_known": int(typed.sum()), "semantic_correct": int(correct.sum()), "roles": fit_roles})
            if args.component_capacity:
                from tetris_vision.components import component_metrics
                per_case[-1]["components"] = component_metrics(output["final"]["components"], target["components"])[0]
            records.append(case.record)
            if persist:
                folder = args.output/(split+"-probe")
                folder.mkdir(exist_ok=True)
                case.frame.image.save(folder/(case.record["sample_id"]+".png"))
                (folder/(case.record["sample_id"]+".json")).write_bytes(case.frame.state.to_json().encode())
        identity = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
        if split in evaluation_identities and evaluation_identities[split] != identity:
            raise ValueError("evaluation cases changed during training")
        evaluation_identities[split] = identity
        if persist:
            (args.output/(split+"-probe.jsonl")).write_text("".join(json.dumps(row)+"\n" for row in records))
        result = {"images": len(per_case), "ancestry_groups": len({row["group_id"] for row in records}), "loss": sum(row["loss"] for row in per_case)/len(per_case),
                "exact_locked_boards": sum(row["exact_locked_board"] for row in per_case),
                "cell_accuracy": sum(row["cell_accuracy"] for row in per_case)/len(per_case),
                "mean_geometry_error_cells": sum(row["mean_geometry_error_cells"] for row in per_case)/len(per_case),
                "exact_known_layers": sum(row["exact_known_layers"] for row in per_case),
                "known_layers": sum(row["known_layers"] for row in per_case), "cases": per_case}
        if args.layer_evidence:
            for key in ("dense_layer_grids_known", "dense_layer_grids_exact",
                        "dense_observed_visible_grids_known", "dense_observed_visible_grids_exact"):
                result[key] = sum(row[key] for row in per_case)
        if args.component_capacity:
            result.update(readable_component_cases=sum(row["components"]["component_state_readable"] for row in per_case),
                          exact_readable_components=sum(row["components"]["exact_readable_components"] is True for row in per_case),
                          supported_components_exact=sum(row["components"]["supported_components_exact"] for row in per_case))
        if args.curriculum in FIT_POLICIES and split == "training":
            result["fit"] = summarize_fit(per_case)
        return result

    initial_train, initial_dev = evaluate("training", 3, True), evaluate("development", args.development_groups, True)
    completed = 0
    recovery_contract = {"catalogue_sha256": archive["catalogue_sha256"], "config": config_json,
                         "seed": args.seed, "learning_rate": args.learning_rate, "weight_decay": .01,
                         "steps": args.steps, "max_side": args.capture_max_side, "development_groups": args.development_groups,
                         "evaluate_every": args.evaluate_every, "checkpoint_every": args.checkpoint_every,
                         "lr_final": args.lr_final,
                         "profile_square": args.profile_square, "threads": 1, "rng": "torch_cpu_only"}
    recovery_contract["native_cache_limits"] = NATIVE_CACHE_LIMITS
    recovery_contract["input_policy"] = "M3-final-capture-v1" if args.component_capacity else "M2-prepared-resize-v1"
    recovery_contract["curriculum"] = args.curriculum
    recovery_contract["type_head_rate_factor"] = args.type_head_rate_factor
    recovery_contract["component_identity_factor"] = args.component_identity_factor
    recovery_contract["component_identity_readout"] = args.component_identity_readout
    recovery_contract["component_endpoint_term"] = args.component_endpoint_term
    recovery_contract["occupancy_factor"] = args.occupancy_factor
    recovery_contract["occupancy_positive_weight"] = args.occupancy_positive_weight
    recovery_contract["occupancy_context"] = args.occupancy_context
    recovery_contract["occupancy_pixel_query"] = args.occupancy_pixel_query
    recovery_contract["pixel_query_level"] = args.pixel_query_level
    recovery_contract["occupancy_read_standardise"] = args.occupancy_read_standardise
    recovery_contract["occupancy_mass"] = args.occupancy_mass
    recovery_contract["occupancy_read_level"] = args.occupancy_read_level
    recovery_contract["occupancy_balance"] = args.occupancy_balance
    recovery_contract["occupancy_detail"] = args.occupancy_detail
    recovery_contract["occupancy_detail_stencil"] = args.occupancy_detail_stencil
    recovery_contract["occupancy_resolution"] = args.occupancy_resolution
    recovery_contract["occupancy_deform"] = args.occupancy_deform
    recovery_contract["occupancy_placed"] = args.occupancy_placed
    recovery_contract["occupancy_column"] = args.occupancy_column
    recovery_contract["occupancy_offset_rows"] = args.occupancy_offset_rows
    recovery_contract["occupancy_normalise"] = args.occupancy_normalise
    recovery_contract["occupancy_margin_hinge"] = args.occupancy_margin_hinge
    recovery_contract["occupancy_anchor"] = args.occupancy_anchor
    recovery_contract["occupancy_quantile_anchor"] = args.occupancy_quantile_anchor
    recovery_contract["occupancy_dense_hinge"] = args.occupancy_dense_hinge
    recovery_contract["occupancy_midpoint"] = args.occupancy_midpoint
    recovery_contract["occupancy_centre"] = args.occupancy_centre
    recovery_contract["occupancy_stack_monotone"] = args.occupancy_stack_monotone
    recovery_contract["occupancy_crossing_threshold"] = args.occupancy_crossing_threshold
    recovery_contract["occupancy_skyline_recovery"] = args.occupancy_skyline_recovery
    recovery_contract["occupancy_tap_span"] = args.occupancy_tap_span
    recovery_contract["geometry_final_only"] = args.geometry_final_only
    recovery_contract["lattice_support"] = args.lattice_support
    recovery_contract["box_extent_weight"] = args.box_extent_weight
    recovery_contract["support_anchor_weight"] = args.support_anchor_weight
    recovery_contract["lattice_residual"] = args.lattice_residual
    recovery_contract["occupancy_lattice_free"] = args.occupancy_lattice_free
    recovery_contract["occupancy_free_stencil"] = args.occupancy_free_stencil
    recovery_contract["occupancy_pair_rank"] = args.occupancy_pair_rank
    recovery_contract["occupancy_pair_gap"] = args.occupancy_pair_gap
    recovery_contract["lattice_board_refine"] = args.lattice_board_refine
    complete_recovery_contract(recovery_contract, json.loads(json.dumps(asdict(config))))
    recovery_contract["freeze_backbone"] = args.freeze_backbone
    recovery_contract["initial_weights_from"] = initial_origin["run"] if initial_origin else None
    if args.accumulation_policy is not None:
        recovery_contract["accumulation_policy"] = args.accumulation_policy
        recovery_contract["images_per_update"] = ACCUMULATION_POLICIES[args.accumulation_policy][0]
    ledger_names = ("curve.jsonl", "training.jsonl", "evaluations.jsonl")
    resume_record = None
    if args.resume_from is not None and args.accumulation_policy is not None:
        # The committed journal prefix and Adam/RNG state are restored into this
        # segment's own directory; any uncommitted suffix in the source is excluded.
        restored = restore_boundary(args.resume_from, args.output, model, optimizer, accumulation)
        completed = restored["optimizer_updates"]
        if (completed >= args.steps or restored["images_seen"] !=
                completed*ACCUMULATION_POLICIES[args.accumulation_policy][0]):
            raise ValueError("accumulated resume requires an unfinished compatible arm")
        prefix = (args.output/"groups.jsonl").read_bytes()
        resume_record = {"source_run": str(args.resume_from.resolve()), "step": completed,
                         "pointer": restored["source_pointer"], "images_seen": restored["images_seen"],
                         "journal_prefix_sha256": hashlib.sha256(prefix).hexdigest()}
    elif args.resume_from is not None:
        recovery = load_training_snapshot(args.resume_from, model, optimizer, contract=recovery_contract)
        completed = recovery["step"]
        if completed >= args.steps or recovery["ledger_prefixes"].keys() != set(ledger_names):
            raise ValueError("resume requires an unfinished compatible stream and all three committed ledgers")
        for name, data in recovery["ledger_prefixes"].items():
            (args.output/name).write_bytes(data)
        # The source cursor is authoritative; later uncheckpointed appends are
        # retained in the parent run and never replayed into the resumed stream.
        for name in ("curve.jsonl", "training.jsonl"):
            records = [json.loads(line) for line in (args.output/name).read_text().splitlines()]
            if [row["step"] for row in records] != list(range(1, completed+1)):
                raise ValueError("snapshot ledger does not match its global step")
        resume_record = {"source_run": str(args.resume_from.resolve()), "step": completed, "pointer": recovery["pointer"],
                         "ledger_prefix_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in recovery["ledger_prefixes"].items()}}
    started_step = completed
    final_step = min(args.steps, started_step+(args.max_new_steps or args.steps))
    latest_snapshot = None
    mode = "a" if resume_record else "x"
    steps_to_run = () if args.accumulation_policy is not None else range(started_step+1, final_step+1)
    with (args.output/"curve.jsonl").open(mode, encoding="utf-8", newline="\n") as curve, (args.output/"training.jsonl").open(mode, encoding="utf-8", newline="\n") as ledger, (args.output/"evaluations.jsonl").open(mode, encoding="utf-8", newline="\n") as evaluations, (args.output/"memory.jsonl").open("x", encoding="utf-8") as memory_log:
        if not resume_record:
            evaluations.write(json.dumps({"step": 0, "training": initial_train, "development": initial_dev})+"\n")
        evaluations.flush()

        if args.accumulation_policy is not None:
            def recipe_for(image, index, view):
                return training_case(image, args.seed, curriculum=args.curriculum,
                                     component_capacity=args.component_capacity).record

            def loss_for(recipe):
                case = replay_case(recipe, split="training", curriculum=args.curriculum,
                                   component_capacity=args.component_capacity)
                prepared, target = prepare(case)
                model.train()
                output = model(prepared.pixels[None], prepared.valid[None])
                loss, _ = joint_loss(output, target, identity_factor=args.component_identity_factor, endpoint_term=args.component_endpoint_term, occupancy_factor=args.occupancy_factor, positive_weight=args.occupancy_positive_weight, geometry_final_only=args.geometry_final_only, box_extent_weight=args.box_extent_weight, support_anchor_weight=args.support_anchor_weight, occupancy_placed=args.occupancy_placed, occupancy_margin_hinge=args.occupancy_margin_hinge, occupancy_anchor=args.occupancy_anchor, occupancy_quantile_anchor=args.occupancy_quantile_anchor, occupancy_dense_hinge=args.occupancy_dense_hinge, occupancy_midpoint=args.occupancy_midpoint, occupancy_pair_rank=args.occupancy_pair_rank, occupancy_pair_gap=args.occupancy_pair_gap,
occupancy_mass=args.occupancy_mass)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("nonfinite accumulated loss")
                return loss

            def on_group(update, state, entry):
                curve.write(json.dumps({"update": update, "images_seen": state["images_seen"],
                                        "optimizer_updates": state["optimizer_updates"],
                                        "mean_loss": entry["mean_loss"], "gradient_norm": entry["gradient_norm"],
                                        "elapsed_seconds": time.monotonic()-started}, allow_nan=False)+"\n")
                for row in entry["images"]:
                    ledger.write(json.dumps({"update": update, "image": row["image"], **row["recipe"]},
                                            allow_nan=False)+"\n")
                curve.flush()
                ledger.flush()

            def on_evaluation(update, state):
                evaluations.write(json.dumps({"update": update, "images_seen": state["images_seen"],
                                              "development": evaluate("development", args.development_groups)})+"\n")
                evaluations.flush()

            def on_progress(update, state, mean_loss):
                progress.update(status="training", update=update, images_seen=state["images_seen"],
                                optimizer_updates=state["optimizer_updates"], loss=mean_loss,
                                elapsed_seconds=time.monotonic()-started)
                atomic_json(args.output/"progress.json", progress)

            if args.resume_from is None:
                create_journal(args.output)
            accumulation_result = run_accumulation_loop(
                run=args.output, policy=args.accumulation_policy, contract=accumulation,
                model=model, optimizer=optimizer, updates_total=final_step, start_update=started_step+1,
                recipe_for=recipe_for, loss_for=loss_for, checkpoint_every=args.checkpoint_every,
                evaluate_every=args.evaluate_every, wall_seconds=args.wall_seconds, started=started,
                on_group=on_group, on_evaluation=on_evaluation, on_progress=on_progress)
            latest_snapshot = accumulation_result["pointer"]
            if accumulation_result["state"] is not None:
                completed = accumulation_result["state"]["optimizer_updates"]
                committed = accumulation_result["snapshotted_images"] == accumulation_result["state"]["images_seen"]
                progress.update(status="training", update=completed,
                                images_seen=accumulation_result["state"]["images_seen"],
                                optimizer_updates=completed, recovery_step=completed if committed else 0,
                                elapsed_seconds=time.monotonic()-started)
                atomic_json(args.output/"progress.json", progress)
            if args.profile_square and accumulation_result["state"] is not None:
                warm_optimizer_backward = accumulation_profile(
                    run=args.output, model=model, optimizer=optimizer, contract=accumulation,
                    policy=args.accumulation_policy, completed=completed, maximum_pixels=maximum_pixels)
                recovery_profile_exact = warm_optimizer_backward["recovery_exact"]
                if not recovery_profile_exact:
                    raise RuntimeError("accumulation boundary snapshot changed model/optimizer/RNG")

        def snapshot():
            for writer in (curve, ledger, evaluations):
                writer.flush()
            return save_training_snapshot(args.output, model, optimizer, step=completed,
                                          contract=recovery_contract, ledgers=ledger_names)

        for step in steps_to_run:
            if time.monotonic()-started >= args.wall_seconds:
                break
            # Both characterization steps must use the maximum shape. Step two
            # includes Adam's moment tensors, allocated lazily by the first step.
            case = training_case(step, args.seed, curriculum=args.curriculum, component_capacity=args.component_capacity, profile_square=args.profile_square)
            prepared, target = prepare(case)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            if args.lr_final is not None:
                # The declared schedule lever: the rate is identical across the segment's
                # own steps, so the two arms differ in nothing but this scalar.
                rate = cosine_rate(args.learning_rate, args.lr_final, step, args.steps)
                for group, base in zip(optimizer.param_groups, scheduled_base_rates):
                    group["lr"] = base*rate/args.learning_rate
            if args.profile_square and step == 2:
                parameters = [p for p in model.parameters() if p.requires_grad]
                initialized = all(
                    p in optimizer.state and all(
                        key in optimizer.state[p] and optimizer.state[p][key].shape == p.shape
                        for key in ("exp_avg", "exp_avg_sq"))
                    and float(optimizer.state[p]["step"]) == 1 for p in parameters)
                if not parameters or not initialized:
                    raise RuntimeError("resource profile requires resident Adam state for every trainable parameter")
                warm_optimizer_backward = {
                    "step": step, "input_pixels": prepared.pixels.shape[-2]*prepared.pixels.shape[-1],
                    "parameter_tensors": len(parameters), "all_parameters_initialized": initialized,
                    "state_bytes": sum(value.numel()*value.element_size() for state in optimizer.state.values()
                                       for value in state.values() if torch.is_tensor(value)),
                    "backward_completed": False}
            output = model(prepared.pixels[None], prepared.valid[None])
            loss, terms = joint_loss(output, target, identity_factor=args.component_identity_factor, endpoint_term=args.component_endpoint_term, occupancy_factor=args.occupancy_factor, positive_weight=args.occupancy_positive_weight, geometry_final_only=args.geometry_final_only, box_extent_weight=args.box_extent_weight, support_anchor_weight=args.support_anchor_weight, occupancy_placed=args.occupancy_placed, occupancy_margin_hinge=args.occupancy_margin_hinge, occupancy_anchor=args.occupancy_anchor, occupancy_quantile_anchor=args.occupancy_quantile_anchor, occupancy_dense_hinge=args.occupancy_dense_hinge, occupancy_midpoint=args.occupancy_midpoint, occupancy_pair_rank=args.occupancy_pair_rank, occupancy_pair_gap=args.occupancy_pair_gap,
occupancy_mass=args.occupancy_mass)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite curriculum loss")
            loss.backward()
            if args.profile_square and step == 2:
                warm_optimizer_backward["backward_completed"] = True
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            completed = step
            item = {"step": step, "sample_id": case.record["sample_id"], "loss": float(loss.detach()),
                    "terms": {key: float(value.detach()) for key, value in terms.items()},
                    "gradient_norm": float(norm), "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "elapsed_seconds": time.monotonic()-started}
            curve.write(json.dumps(item, allow_nan=False)+"\n")
            ledger.write(json.dumps({"step": step, **case.record}, allow_nan=False)+"\n")
            curve.flush()
            ledger.flush()
            if step == started_step+1 or step % 16 == 0 or step == final_step:
                memory_log.write(json.dumps({"step": step, "input_shape": list(prepared.pixels.shape),
                                             "phase": "post_update", **bounded_job_snapshot()})+"\n")
                memory_log.flush()
            progress.update(status="training", step=step, loss=item["loss"], elapsed_seconds=item["elapsed_seconds"])
            atomic_json(args.output/"progress.json", progress)
            print(json.dumps(item), flush=True)
            if step % args.evaluate_every == 0:
                evaluations.write(json.dumps({"step": step, "development": evaluate("development", args.development_groups)})+"\n")
                evaluations.flush()
            if step % args.checkpoint_every == 0 or step == final_step or args.profile_square:
                latest_snapshot = snapshot()
                progress["recovery_step"] = completed
                atomic_json(args.output/"progress.json", progress)
            if args.profile_square and step == 1:
                # Characterize actual recovery allocation as well as warm
                # backward. Deep copies intentionally make this a conservative
                # peak; the native cap remains authoritative.
                expected_state = copy.deepcopy((model.state_dict(), optimizer.state_dict(), torch.get_rng_state()))
                recovery = load_training_snapshot(args.output, model, optimizer, contract=recovery_contract)
                recovery_profile_exact = recovery["step"] == 1 and equal_tree(expected_state, (model.state_dict(), optimizer.state_dict(), torch.get_rng_state()))
                del expected_state, recovery
                if not recovery_profile_exact:
                    raise RuntimeError("training snapshot changed model/optimizer/RNG")
        if args.accumulation_policy is None and completed > started_step and (latest_snapshot is None or progress.get("recovery_step") != completed):
            latest_snapshot = snapshot()
            progress["recovery_step"] = completed
    final_train, final_dev = evaluate("training", 3), evaluate("development", args.development_groups)
    verify_source(code, archive["catalogue_sha256"])
    if file_digest(initial_path) != initial_sha256:
        raise ValueError("saved initial tensors changed during execution")
    accumulation_metadata = None
    if args.accumulation_policy is not None:
        size = ACCUMULATION_POLICIES[args.accumulation_policy][0]
        accumulation_metadata = {
            "policy": args.accumulation_policy, "images_per_update": size,
            "images_seen": completed*size, "optimizer_updates": completed,
            "contract": accumulation, "journal_sha256": file_digest(args.output/"groups.jsonl"),
            "latest_boundary": latest_snapshot, "resumed_from_update": started_step,
            "images_this_segment": (completed-started_step)*size}
    metadata = {"trained_steps": completed, "catalogue_sha256": archive["catalogue_sha256"],
                "initial_weights_sha256": initial_sha256,
                "supervisor_script_sha256": supervisor_sha, "system_pressure_guard": "GetPerformanceInfo/0.2s",
                "runtime_source_sha256": source_digest(), "model_kind": args.model,
                "component_capacity": args.component_capacity, "input_policy": recovery_contract["input_policy"],
                "freeze_backbone": args.freeze_backbone, "initial_weights_from": initial_origin,
                "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "stream_ledger_sha256": file_digest(args.output/"training.jsonl"), "cpu_only": torch.version.cuda is None,
                "resume": resume_record, "latest_training_snapshot": latest_snapshot,
                "driver_source_path": str(Path(__file__).resolve()),
                "native_cache_limits": NATIVE_CACHE_LIMITS,
                "recovery_contract": recovery_contract, "updates_this_run": completed-started_step,
                "accumulation": accumulation_metadata}
    checkpoint = save_checkpoint(args.output/"checkpoint", model.eval(), metadata)
    restored, _ = load_checkpoint(args.output/"checkpoint")
    example, _ = prepare(sample(0, 0, "development"))
    with torch.no_grad():
        expected = model(example.pixels[None], example.valid[None])["final"]
        actual = restored.eval()(example.pixels[None], example.valid[None])["final"]
    if not equal_tree(expected, actual):
        raise RuntimeError("saved checkpoint changes a final output tensor")
    status = "complete" if completed == args.steps else "step_partial" if completed == final_step else "wall_partial"
    summary = {"status": status, "steps": completed, "requested_steps": args.steps,
               "resource_profile_version": 3, "training_recovery_profile_exact": recovery_profile_exact,
               "warm_optimizer_backward": warm_optimizer_backward,
               "profile_square": args.profile_square, "max_input_pixels_seen": maximum_pixels, "max_side": args.capture_max_side,
               "job_ceiling_bytes": ceiling, "development_groups": args.development_groups, "config": config_json,
               "threads": 1, "workers": 0, "prefetch": 0, "tensor_cache_bytes": 0,
               "fresh_parent_groups": distinct_training_groups(
                   completed*(ACCUMULATION_POLICIES[args.accumulation_policy][0]
                              if args.accumulation_policy and not args.profile_square
                              else 1),
                   args.curriculum, profile_square=args.profile_square),
               "curriculum": args.curriculum,
               "initial_training_probe": initial_train, "initial_development": initial_dev,
               "final_training_probe": final_train, "development": final_dev, "checkpoint_round_trip_exact": True,
               "checkpoint_sha256": checkpoint["weights_sha256"], "elapsed_seconds": time.monotonic()-started,
               **({"accumulation_policy": args.accumulation_policy,
                   "images_per_update": accumulation_metadata["images_per_update"],
                   "images_seen": accumulation_metadata["images_seen"], "optimizer_updates": completed,
                   "accumulation_journal_sha256": accumulation_metadata["journal_sha256"],
                   "optimizer_updates_per_second": completed/max(time.monotonic()-started, 1e-9),
                   "resumed_from_update": started_step, "resume": resume_record}
                  if accumulation_metadata is not None else {}),
               "claim": "Finite synthetic curriculum or resource characterization, not complete-state or real-image acceptance.", **metadata}
    atomic_json(args.output/"summary.json", summary)
    progress.update(status=summary["status"], step=completed, summary="summary.json", checkpoint="checkpoint/manifest.json")
    atomic_json(args.output/"progress.json", progress)
    print(json.dumps({"status": summary["status"], "steps": completed, "max_input_pixels_seen": maximum_pixels,
                      "checkpoint_round_trip_exact": True, "catalogue_sha256": archive["catalogue_sha256"]}), flush=True)
    return 0 if summary["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
