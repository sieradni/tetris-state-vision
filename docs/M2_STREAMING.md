# Bounded streaming curriculum

`scripts/train_m2_stream.py` implements the next synthetic data experiment without
changing the M2 model, loss, preprocessing, renderer or visibility rules. The
first resource characterization passed; a substantial curriculum study has not
run yet. The pilot remains distinct from the unprofiled accuracy configuration.

Each group contains a new empty/sparse/dense state, a variable source board
position/scale, clean and no-grid appearances, and one visible single-cell
intervention. The intervention avoids active/ghost cells and stays in the same
ancestry group. Training/development use disjoint seed ranges. Vertical preview
flow is the current data scope; this is not broad renderer/client coverage.
Inputs preserve aspect and use at most 512×512 padded pixels. The sampler and
trainer require all 200 source/input locked cells to remain readable.

Frames and tensors are generated one at a time. There are no cached tensors,
prefetch or worker processes. Each optimizer step records the sample recipe,
pixel/state identities, loss terms and gradient norm. Fixed training/development
probes are saved separately. Matched appearances are dependent observations,
and no synthetic metric is a real-image or complete-state accuracy claim.

Before heavy imports, the worker verifies membership in the named native Job.
It then copies a bounded source archive into the new run's `code/` directory:
package modules, five explicit runner helpers, project metadata and lock file.
The catalogue has per-file SHA256 values, covers nested modules, and is pinned
in checkpoint/run metadata. Imports explicitly bind to that archive. Parent
source edits do not affect the running recipe. Archive mutations or extra source
modules fail verification. Source archives, weights and generated probes stay in
ignored local `runs/`; curated receipts retain their identities.

Resource admission is explicit. `--model pilot` uses the 647,098-parameter
configuration; `--model accuracy` selects the larger research configuration.
Neither silently substitutes for the other. First run exactly two optimizer steps
with `--profile-square`, which repeats a real fully visible 512×512 source
frame. Version2 verifies full-size Adam moments for every trainable parameter
before the second forward, and records their bytes and the completed backward.
Adam lazily allocates moments during the first update, so a cold one-step probe
does not cover warm training memory (V-011). A longer run requires `--admission PATH` naming a completed warm square probe
with the same source catalogue, model configuration, CPU settings and memory
ceiling, and enough development groups. Wall-partial probes and failed native
receipts cannot admit training. Limits: 1..4096 steps, at most 1800 inner seconds,
one CPU thread, 3..12 development ancestry groups. The outer launcher still
supplies independent wall, memory and declared work caps.

The historical one-step command was (not current admission):

```text
run_bounded.py --memory-mib 1024 --wall-seconds 150 --work-items 1
--receipt runs/m2-stream-probe-job.json -- .venv/Scripts/python.exe
scripts/train_m2_stream.py --output runs/m2-stream-probe --model pilot
--steps 1 --profile-square --wall-seconds 110 --development-groups 6
```

Launch longer commands hidden/detached with logs and a checkpoint entry. The
actual probe completed in 9.717 s, raw peak 549,154,816 B within 1 GiB. It reached
262,144 input pixels and saved/reloaded **every final output tensor exactly**.
Fourteen fixtures cover sampler contracts, source mutation/parent edits/replay
and admission refusal. The curator re-rendered the actual training sample and
all 27 saved probe images/annotations using the archived sampler, with exact
identities. Evidence: `docs/evidence/m2-stream-probe/scope.json`.

An archived runner can be replayed directly from `RUN/code/scripts/` through the
current guarded supervisor; do not invoke a historical pre-V-010 supervisor.
Source capture accepts the archived package layout as
well as the repository layout. Dependencies must still match its archived lock.
`scripts/record_stream_run.py` verifies a completed run's finite ledgers, source
catalogue, generated pixels/labels and weight identity before curating evidence.
All output paths are exclusive. Inner-wall partial runs keep partial status and
are not accepted by this completion curator.

The system pressure guard now passes 13 pressure/native/admission checks; V-010
is CLOSED. See [RESOURCE_GUARD.md](RESOURCE_GUARD.md). The old probe cannot admit
the strengthened current driver. Four focused version2 admission fixtures pass
in docs/evidence/v011-protocol/; actual warm resource characterization passed:
docs/evidence/m2-stream-warm-probe/scope.json. Two square steps,142 initialized
parameter tensors,5,177,352B optimizer state;9.553s /594,829,312B raw peak under
640MiB, all-final-tensor reload exact. V-011 CLOSED. Catalogue
42148a03905297502d6a36e02622087f808b8c36cc10883474a4b1bd84b8770b
admits only matching sources/settings/ceiling. The completion curator verifies
warm protocol and replays2 samples plus27 probes; fixture evidence has separate
named source scope. Next: measure fresh-state learning. Profile the accuracy model
separately. Queue/hold joint modeling, broad data, real reviewed accuracy and
release gates remain open.
