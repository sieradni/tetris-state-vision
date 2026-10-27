# Conditional lattice diagnostic

This asks whether supplying the correct lattice helps a saved model read its
cells and independent active/ghost layers. It deliberately supplies synthetic
labels in a second forward. Its scores describe that intervention on that model;
they are not automatic parsing accuracy, a model-capacity bound or a correction
used by the runtime. No weights, training labels or production output change.

`probe_stream_geometry.py` format2 accepts a complete M2 artifact or an explicit
`--allow-partial` controlled saved result. The native training receipt and exact
checkpoint must agree; a pressure stop without a final inference artifact is
ineligible. Actual architecture/configuration/source/weights and original
global cursor, target and partial status remain in the report. Optional M3
component checkpoints are refused by this diagnostic. Their support is separate.

For the current nine training and eighteen development probes, budget54 forwards
(ordinary plus conditional per image). Each recipe, raster, annotation and frozen
callable origin is checked, ordinary metrics must reproduce the saved artifact,
and supported board geometry is required. Oracle geometry error is deliberately
excluded: reporting zero after supplying that geometry would be tautological.

Serialize after training's entire process tree terminates. For a larger saved
artifact, expected peak below850MiB based on measured ordinary-forward memory,
one weight clone and retained paired outputs; enforce1GiB/120s/work54 with native
Job and OS pressure guards. Use a fresh output, receipt, hidden logs, checkpoint
entry and exact PID verification. Example argument suffix for the separately
bounded runner (only after s4 has a controlled final artifact):

```text
.venv/Scripts/python.exe scripts/probe_stream_geometry.py --run m2-stream-accuracy4096-s4 --output runs/m2-stream-accuracy1664-geometry-r1 --allow-partial
```

Do not use the example's output label as a step assertion; preserve the actual
artifact cursor. The parent target remains4096 until reached. A changed resource
window or incomplete artifact does not authorize bypassing admission.

The collector `record_stream_geometry.py --name <output-name>` runs under
128MiB/30s/work54 (expected below96MiB). It checks full native command/output,
finite work/memory, actual checkpoint/status, unique cases and ancestry, every
ordinary metric and recomputed ordinary/conditional totals. It preserves raw
input summary/probe recipes/receipt and source/weight manifests with hashes.
Historical pilot geometry reports keep their old scope; new format2 evidence
does not retroactively upgrade them. Metadata fixtures are distinct from actual
paired forwards. V-024 closes only on named native replay and curation evidence.

Ten scoped metadata checks pass in.113s,35,180,544B under128MiB/30s on eaf5bee:
`docs/evidence/geometry-evidence-contracts/scope.json` pins18 source/fixture
files. They include the preserved pilot's negative oracle effect, changed
aggregates and native command, missing/duplicate cases, explicit partial
admission, wrong architecture and unsupported/nonfinite targets. Historical
fixture data is adapted only in memory; it is not a new model evaluation.

Actual saved1664 format2 replay/curation passes at
`docs/evidence/m2-stream-accuracy1664-geometry-r1/scope.json`:54 forwards in23.894s,
585,072,640B under1GiB;collector.288s,64,712,704B under128MiB. All ordinary
metrics reproduce and weights remain unchanged. Both modes have0/18 exact
development boards;cell accuracy decreases from.920278 to.904444 with the
supplied lattice,and exact known layers from2/36 to0/36. This shows that this
intervention alone does not repair this checkpoint. It neither identifies a
capacity ceiling nor proves which architecture/training change will work.
