# M39: the same cell read at both placements

Status: **predeclared before any M39 scoring run.** Mechanism, widths, protocol, the unchanged gate, the
promotion bar and both branch interpretations are fixed here.

## The hypothesis this pass exists to test

The campaign's regression is stark and it is not about features or width: every arm that samples the raster
at the capture's fixed support lattice ranks below `m30-level-calibrate-r5`, which samples at *its own
predicted lattice* (M34 `0.883260` AUC, M35 `0.945596`, M36 `0.886350`, M37 `0.919512`, M38 `0.895041`
against the winner's `0.954529`; best swept `0.385925`-`0.589286` against `0.689775`). Three feature
families and a doubled width are refused, so capacity is not the explanation.

The winner's arm has one thing no support-lattice arm has: its samples move with the geometry the decoder
believes, so its features and its placement are the same object. The support-lattice arms broke that
identity in order to measure placement, and each of them then ranked below the arm that kept it.

The dual-placement read makes that difference available to the read instead of choosing between the two: if
the consistency between the two placements is what the winner's arm exploits implicitly, giving the read
both halves should recover part of the ranking; if the ranking loss is simply that the support lattice
samples the wrong pixels per board, the halves carry the disagreement but no information the read can use,
and the arm will land with the other fixed-lattice arms.

## The mechanism, exactly (form 13)

`--occupancy-pixel-query 13`, one declared form, one head block, no second selector:

- **Half A**: the `30` statistics (nine sub-cell samples x three channels + three per-channel spreads) at
  the **model's own predicted lattice** for the stage, exactly as the published read samples.
- **Half B**: the same `30` statistics at the **capture's support lattice**
  (`pixel_support_lattice`: the bounding box of the valid pixels, mapped bottom-first).
- **Fusion**: concatenation, `60` inputs, through the unchanged hidden projection
  `Linear(60, 64) -> SiLU -> Linear(64, dimension)` with a zero-initialised output, so step zero is the
  published function and no output parameter is added. Against the `30`-statistic form the arm adds
  `30*64` hidden parameters and nothing else.
- **Boundary**: the consumer's fixed global `0.0`. Nothing is fitted, no label, no prevalence, no scored
  cell, no threshold, and neither half reads anything but the raster and geometry the model already has.

## Protocol

One profile, one 64-step study, one 36-board probe under the `m39-dual` owned family (1,152 MiB study,
640 MiB probe), scored by `scripts/m37_stencil_read.py` with `--inputs 60 --instrument m39-dual-r1
--predeclaration docs/M39_DUAL_PLACEMENT.md` on the same 2,986 cells and both fixed label vectors, the
winner re-scored in the same pass.

## Gate thresholds, unchanged

per-cell `0.70` on both fixed label vectors; ranking (AUC `>= 0.90` and best swept `>= 0.70`); attribution
over the unchanged control `>= 0.05`.

## The deciding bar, fixed before scoring

New operating point only if, against the winner's re-scored numbers:

- F1 at `0.0` `>= 0.372881`, and
- best swept F1 `>= 0.689775`.

**A refusal means** the presence of the predicted-lattice half is not sufficient on its own: the read's
advantage in the winner's arm is not transferable by handing it both geometries, and the campaign must state
`B in this form` in its own terms - the fixed-support arms' deficit is a property of *how* a read is tied to
geometry, not of which features it sees.

## Branch interpretations, pre-committed

- **PASS**: the arm becomes the operating point; re-run its probe and reproduce weights and logits (not the
  earlier all-zero delta by assumption), and record the next structural step.
- **FAIL**: diagnose in measured terms whether the placement-consistency signal is present but
  insufficient, absent, or swamped - using the false-positive counts, margins and band figures exactly as
  M35-M38 did - and do not attribute the loss to capacity. Then record the sharpest structural statement the
  accumulated evidence supports and one structural next step that is not a width or a tuning variant.

## Receipts

`runs/m39-dual-{profile,study,probe}` and `docs/evidence/m39-dual-placement-r1/{scope,diagnosis}.json`.
Refused attempts keep their receipts; only completed runs are quoted.
