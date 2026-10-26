# Source-frozen, resumable streaming curriculum

The current driver generates a new empty/sparse/dense state and variable board
position/scale per ancestry group. Each group has clean, no-grid and visible
single-cell-change views. Training/development use disjoint seed ranges. The
current scope is a fully visible single affine board with vertical previews,
not broad real screenshots or complete-state accuracy.

Frames/tensors are generated one at a time. No tensor cache, prefetch or workers
are retained. Inputs preserve aspect and use at most512x512 padded pixels; all
200 locked cells must remain image-supported. Model inputs are pixels/validity
only. Recipes, pixel/state hashes, loss terms and gradient norms are recorded
per update. Matched views share ancestry and are dependent observations.

Before heavy imports the process verifies its named native Job. The bootstrap
copies a bounded archive (package modules, seven runner helpers, project and
lock metadata), verifies hashes, then loads trainer main and helpers from that
archive. Parent edits cannot change running model/sampler/driver bytecode or
traceback source. Per-file1MiB/aggregate16MiB/module-count bounds apply. Source
catalogue and inference/recovery weights retain their hashes and local paths.

Resource admission version3 requires two real512-square updates. The second
backward has full resident Adam moments. The first update also saves/reloads
training state and verifies model/optimizer/RNG equality. A longer run requires
`--admission RUN` with the same catalogue/model/config/CPU envelope/ceiling and
at least as many development groups. Old/cold/partial/failed profiles refuse.
The pilot647,098-parameter model and larger accuracy configuration are explicit;
profile each independently. A pilot result cannot select the release model.

CPU libraries receive one-thread defaults before import. The training worker
requests a16-entry oneDNN primitive cache; controlled native logs demonstrate
cache eviction. Raw Job peaks are sampled every16updates. The kernel640MiB
ceiling and system pressure guard remain authoritative. A two-step square probe
alone did not characterize long varied-shape memory: the old768 study failed
at565 updates with no final weights (V-013). Its raw evidence is retained under
docs/evidence/v013-failure/. A fresh768-study memory gate is separate.

Every64updates and at a controlled exit, the driver commits a training snapshot:
model, AdamW state, Torch CPU randomness, module modes, global step, exact recipe
and three ledger-prefix identities. Latest pointer publishes last. Resume with
`--resume-from RUN` into a fresh output. `--steps` is the global target;
`--max-new-steps` optionally bounds one segment. Segments keep `step_partial`,
wall exits keep `wall_partial`; neither is a completed learning run.
See TRAINING_RECOVERY.md for identity/refusal and finite artifact bounds.

The native2+6+3+3 recovery gate passes: continuous6 and3+resume3 match recorded
losses/gradients/data/evaluation, every model/optimizer tensor and CPU RNG.
The curator replayed17 recorded training rows (including resumed prefixes) and
72 saved probe rasters/annotations. Source/artifact scope:
docs/evidence/m2-stream-recovery-gate-r2/scope.json. Twenty-nine independent
resource/atomic/source/recovery/admission checks are pinned separately at
docs/evidence/v015-contracts/scope.json; these are not a whole-suite claim.

Current six-development-group warm characterization:
docs/evidence/m2-stream-resume-probe/scope.json,18.986s inner/20.422s outer,
549,130,240B raw peak under640MiB, recovery plus every-final-tensor parity.
Catalogue121fd05a... admits matching sources/settings. The768 study completed:
docs/evidence/m2-stream-learn768-r3/analysis.json,458.494s inner,519,782,400B
raw peak under640MiB. Exact565 prior recorded loss/gradient/data prefix and
2,748,416B second-half peak growth pass the predeclared32MiB gate; recovery768
and all-final-tensor reload exact. Full768 sample+27 probe replay passes. V-013
is CLOSED. Development0/18 exact boards and1.85-cell geometry still fail the
learning objective. No failed/partial result becomes complete by curation, and
no synthetic result establishes real screenshot accuracy.

All output/receipt paths must be new. Use hidden detached launches with logs,
explicit RAM/wall/update caps and the current guarded supervisor. Never invoke
an archived pre-pressure supervisor. Preserve all run/snapshot evidence.
`scripts/record_stream_run.py` replays recipes/probes and verifies inference and
recovery artifacts before curating a completed run. Older one-step receipts
remain historical; full M2 learning, M3 state, real studies and release are open.
