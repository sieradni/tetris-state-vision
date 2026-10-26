# State and annotation policy — schema 1.0

`VisionState` is the authoritative rich JSON format. A latent renderer record
is stored in another file and is never passed off as an image-supported label.
The schema is enforced by state.py constructors and the `validate` CLI.

- `board.rows[0]` is the bottom row; columns run left to right. Each cell has
  occupancy `true`, `false` or `null`. `null` means unknown. Semantic type is
  I/O/T/S/Z/J/L/G or null, independent of known occupancy. G means known garbage;
  null on an occupied cell means unknown type. Literal RGB is not a fumen label.
- Lattice origin is the projected bottom-left boundary in image pixels. Cell
  center is `origin + (column+.5)*u + (row+.5)*v`. Image y increases downward,
  canonical row increases upward. Missing row/column anchors conservatively
  produce null geometry; v1 does not extrapolate cropped coordinates as known.
- Active and ghost have separate status, semantic identity and canonical four-
  cell footprint. Unobserved components have null identity and no latent cells.
  Locked occupancy is never derived by subtracting either component.
- Queue items are future previews, preserving every unknown middle index.
  Visibility is visible/absent/cropped/unknown. `end_observed` refers to the
  visible panel's endpoint, never the hidden future randomizer. Absent queue
  and a visible panel with zero slots are different labels. Capacity: 64 output
  slots; initial renderer capacity: 10. Exceeding capacity raises an explicit
  error; there is no silent truncation.
- Hold distinguishes observed piece, confirmed empty panel, absent, cropped
  and unknown. Hold availability and lock timing are not inferred. Solver
  conversion requires caller-supplied hold availability and refuses missing or
  unknown cells, active piece, queue or hold. Active is included exactly once.
- `accepted` requires supported geometry and known occupancy/components. It
  permits unknown per-cell semantic types; colored scoring separately declares
  type identifiability. `needs_review`, `no_board`, and `unsupported` are explicit.
  Model ID is null for annotations and synthetic labels: no weights exist yet.

## Renderer identifiability policy and limits

The initial renderer is a label-integrity pilot, not adequate training coverage.
It supports one 10x20 axis-aligned board at a fixed source layout, bottom-first
spawn-margin poses, flat/bevel/outline skins, independent layer alpha, simple
backgrounds, absent panels, crops, noninteger resize and limited blur. Queue is
vertical and unrotated; no wrapped/horizontal layouts, photographs, multi-board
scenes, animation, perspective or arbitrary occluders yet. Shadows/glow, linear
blending and independent preview palettes are future renderer families. Compositing
currently uses Pillow's straight-alpha sRGB behavior; masks contain per-layer
alpha before occlusion, not proof of final semantic observability.

Known pilot families use distinct skin conventions for locked/active/ghost.
That convention is not asserted for real clients. When layer cues are disabled,
matching-pixel active-versus-locked scenes both produce the same unknown labels.
Monochrome locked fragments retain occupancy with unknown types. Remapped types
without observed queue/hold/active shape support remain unknown. Canonical
unpermuted palettes are a declared identifiable renderer family, not a runtime
color-threshold rule. Zero/weak alpha, small scale and strong blur conservatively
mask targets. Partial minos/icons are unknown rather than completed from memory.
The thresholds are pilot annotation rules, not calibrated human recoverability.
Real-frame ambiguity requires independent review before training or evaluation.

## Evaluation and split policy

`ExactSolverState` requires every locked occupancy, active type/footprint,
ordered queue/status/endpoint, hold and player to match jointly. Geometry has
separate integrity tests; a normalized geometry metric is still to implement.
`ExactAllVisualLayers` additionally checks ghost. `ExactColoredState` additionally
checks every occupied semantic type on the predeclared identifiable subset.
Unknown type coverage is reported; it never becomes a successful colored guess.
Abstentions and missing predictions remain failures in the readable denominator.
Ambiguous/no-board/unsupported inputs are counted separately, including unsafe
acceptance; no-board recognition and per-field error taxonomy need real fixtures.

Synthetic counterfactuals of one latent state share a deterministic ancestry
group and split (80/10/10 hash buckets). Tiny demos are not expected to populate
all splits. Future real data needs session/video/replay/ancestry and held-out
client/skin/renderer families together; this initial split alone does not achieve
family holdout. No user image or correction becomes a label without review.
Human pilot/baseline, cluster-aware uncertainty, calibration and sealed final test
are pending. Image-level Wilson intervals are diagnostic only, with independence
limits explicitly returned. Self-comparison is never a model accuracy result.
