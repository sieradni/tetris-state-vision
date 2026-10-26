# Queue/hold target contract — verification pending

`component_targets.py` is a pure projection of validated schema 1.1 observation
fields for a future joint decoder. No component head, loss or runtime decoding
is wired yet. Eight fixtures are written but have not run while the host guard
is parking jobs. This is a reviewed source draft, not a green M3 milestone.

Queue token IDs represent seven piece types and EOS; -1 is masked uncertainty.
There are at most 64 visible item positions plus one extra EOS position.
Unknown identity does not erase the slot or shift later indices.

| Observed anchors | Indexed targets |
|---|---|
| Start unseen | All absolute slot/token/presence targets masked; count unknown |
| Start seen, end unseen | Supported visible prefix only; unseen tail masked; no EOS/count |
| Both seen | Supported prefix plus EOS at its observed length; count known |
| Panel absent | No EOS or known-empty count; separate absence status |
| Visible empty panel | EOS at zero with known count zero |

Presence and slot visibility are independent of piece identity. A middle slot
may be known present with unknown/cropped identity. Padding after an observed
endpoint is not trained as the game's hidden future. A 64-item panel can put EOS
at position64; an undersized configured capacity refuses an aligned prefix
instead of dropping known items. Unseen-start spans do not acquire an absolute
count merely because the schema retains their visible relative order.

Hold has independent observed/empty/absent/cropped/unknown status and conditional
piece identity. Unknown type is null, never the first piece class. Identified
model outputs are refused as targets; independent corpus provenance, permission
and review checks remain the training boundary's responsibility.

The paired-raster fixture changes wholly hidden queue length/types and hold
identity while keeping the crop byte-identical. Targets must remain equal, with
no EOS/count or hidden identities. It extends the standing V-001/V-002/V-005/
V-006 visibility instrument. Other fixtures cover unknown middle slots, missing
starts/tails, absence versus visible-empty, all hold statuses, capacity64 and
model-output refusal.

Next: run these eight fixtures inside a 128 MiB /30 s Job when the real host
guard admits it. Then add bounded shared role/index queries and joint losses,
preserving legacy checkpoint behavior and keeping all inference review-only
until quality/calibration gates run. Model architecture work should follow the
verified target contract, not assume the unrun tests are green.
