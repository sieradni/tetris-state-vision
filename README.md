# Tetris State Vision

An independent image-to-state model research project. It includes state contracts,
exact-state scoring, synthetic scenes and a scratch lattice-query model with
optional joint queue/hold heads.
**There is no released complete screenshot parser or validated model yet.**

The [latest matched learning pilot](docs/LAYER_LEARNING_PILOT.md) completed two
768-update studies and a192-image disjoint-ancestry diagnostic. The optional
layer-evidence mechanism did not pass the joint learning gate and remains off
by default. Resource/recovery checks passed; real-image accuracy remains untested.
The [balanced32-image fit](docs/ALL_CLASS_FIT.md) completed1,024 updates and fails
its fit gate. Independent replay finds32/32 exact locked boards but12/56 correct
visible piece labels; positions and rotations are correct, but types collapse
to Z/J/L. The completed [order-only comparison](docs/INTERLEAVED_FIT.md) reaches
30/56 visible labels and28/32 boards and also fails the joint-fit gate.

Requires Python 3.11+ and uv. On Windows, these commands install the locked
dependencies and generate three ancestry groups (15 images). Each receipt/output
path must be new; an existing run is preserved. The launcher refuses other OSes
until an equivalent kernel memory guard is provided.

```powershell
python scripts/run_bounded.py --memory-mib 512 --wall-seconds 120 --work-items 1 --receipt runs/install.json -- uv sync --frozen
python scripts/run_bounded.py --memory-mib 256 --wall-seconds 60 --work-items 15 --receipt runs/demo.json -- .venv/Scripts/python.exe -m tetris_vision generate --count 3 --output runs/demo
```

Launch operations that may exceed 30 seconds with a hidden `Start-Process` and
redirected logs as recorded in docs/CONTINUATION_STATE.md. The supervisor enforces
RAM and wall limits; the child enforces finite generation count. A generic
`--work-items` declaration does not bound arbitrary child code.

Generated images, alpha masks, supported state and separate latent scene records
live beside a checksum manifest and completion sentinel. `validate state.json`
checks the contract. `evaluate --manifest manifest.jsonl --predictions answers.jsonl`
scores predictions of the form `{"sample_id":"...","state":{...}}`; missing
answers fail on readable images. Invoke those through the same bounded launcher.

The CLI defaults to the diverse profile (variable source geometry and queue flow).
`--profile foundation` retains the original five-family pilot configuration.

The variable-layout preview is [examples/layout-diversity.png](examples/layout-diversity.png),
with explicit [queue layouts](examples/queue-flows.png). Its 67-check milestone
receipt is in docs/evidence/m1/layout.json. The later canonical-frame repair is
verified by 69 checks under docs/evidence/v006/ (current contract gate).

The first preview is [examples/montage.png](examples/montage.png); representative
complete and cropped states are in examples/. The state API is
`tetris_vision.state.VisionState`; strict conversion is
`tetris_vision.adapter.to_solver_input(state, hold_available=...)`.

The original foundation receipt is docs/evidence/foundation.json (f5ed97f's
sources, preserved as historical evidence). No engine
runtime, detector, downloaded weights or existing user images are dependencies.

Optional neural work uses `uv sync --frozen --extra train` through the bounded
launcher (1 GiB / 600 s / one sync). It installs CPU-only PyTorch on Windows.
`tetris_vision.model.TetrisStateNet` exposes the M2 raw heads; preprocessing is
`tetris_vision.preprocess.prepare_image`. The accuracy configuration is distinct
from `ModelConfig.pilot()`, which is for resource, numerical and bounded learning pilots.
Source/gradient/checkpoint gates: 85 checks, docs/evidence/m2-numerics/numerical.json.
Learning, queue/hold joint decoding, real-image evaluation and release gates remain
open; raw head diagnostics do not establish complete-state accuracy.

The first 16-step CPU optimizer pilot is recorded under docs/evidence/m2-learn16/
with [predicted-lattice diagnostics](examples/m2-learn16.png). Its loss decreased,
but geometry remained imprecise and development exact boards were 0/6. The weights
are retained locally at runs/m2-learn16/checkpoint/ with a checksum manifest.
See docs/M2_LEARNING_ACCEPTANCE.md before interpreting or extending this result.

The controlled 128-step study reproduces the first 16 steps exactly. Training
loss decreases, but only the empty training boards are exact and development
boards remain 0/6. [Per-cell errors](examples/m2-learn128-locked-errors.png) and
docs/evidence/m2-learn128/comparison.json record this failure to fit nonempty
boards precisely; no screenshot accuracy is established.

The subsequent time-limited study saved step511/512: all six training boards
match, but ghost pose/geometry still fail the declared fit criterion and
development remains 0/6. Its partial status and errors are retained under
docs/evidence/m2-learn512/; it is not a completed 512-step or quality gate.

The next [streaming curriculum](docs/M2_STREAMING.md) generates fresh states and
positions without caching tensors. Runs execute a verified source archive,
require matching resource admission and save resumable model/Adam/RNG state.
Twenty-nine scoped contract checks and an exact six-update versus3+resume3
native gate pass. A complete768-update study now passes the unchanged640MiB
memory gate, including its exact565-record failure prefix and2.6MiB second-half
peak growth. Development still has0/18 exact boards and1.85-cell mean geometry
error. [Evidence](docs/evidence/m2-stream-learn768-r3/analysis.json) establishes
resource/recovery progress; M2 learning and whole-state quality remain open.

The larger7.69-million-parameter candidate also passed its1GiB resource profile
and completed a128-update diagnostic using exact93+35 recovery. Its four exact
development boards all have no locked blocks; nonempty boards remain0/14 and
active/ghost layers0/36. [Recorded comparison](docs/evidence/m2-stream-accuracy128-r2/analysis.json)
does not establish convergence or real-image accuracy.

The separate fixed4096-update board-model study is complete. Its development
probes have0/18 exact boards and0/36 exact active/ghost layers, with2.01544-cell
geometry error. [Current evidence](docs/evidence/m2-stream-accuracy4096-s10/scope.json)
replays all4,096 training recipes and27 saved probe images and retains native controller/
child/curator receipts. Earlier pressure interruptions and exact optimizer/data
recovery remain recorded separately. The original source and optimizer history
are preserved; this study is not extended. Adequate learning remains open.
The [read-only type-feature diagnostic](docs/LAYER_FEATURE_DIAGNOSTIC.md) is
prepared for after the baseline finishes; its analysis-only checks pass, while
neural feature replay remains pending.

Optional joint queue/hold heads are available with `component_capacity=1..64`;
zero retains the board-only model. Shared role/index queries and masked losses
have numerical and compatibility checks, including unknown-slot and hidden-tail
cases. Pilot and larger configurations now pass bounded warm resource/reload
profiles, and continuous6 versus3+resume3 pilot training recovers exactly.
[M3 scope](docs/M3_COMPONENT_TARGETS.md) records the synthetic final-capture
policy and remaining learning/runtime work; the public inspection API still
leaves queue and hold unknown. Resource and recovery checks establish no accuracy.

The inspection API is `ImageParser.from_checkpoint(path).parse_image(image)` in
`tetris_vision.runtime`. It uses the full image, always returns needs_review (or
unsupported), and leaves queue/hold unknown pending verified component decoding.
Run the guarded CLI with
scripts/inspect_m2_checkpoint.py; its real 16-step-artifact call and seven focused
contracts are scoped in docs/evidence/m2-runtime/. The resulting JSON is there
as state.json. This path is for inspection, not automatic solver input.

The design source is archived in docs/research/Tetris_Vision_Research.md.
See docs/ROADMAP.md for milestones and docs/ANNOTATION.md for label semantics.

Local real-image preparation and audit are described in [docs/CORPUS.md](docs/CORPUS.md).
Imports stay drafts until independently reviewed and permitted. Hashes, related
image groups and externally pinned held-out snapshots protect benchmark identity;
these structural checks are not evidence of label correctness or model accuracy.
The user-approved 166-image folder is retained as an [unlabeled stress set](docs/REAL_STRESS_SET.md),
with seven local overview pages and no board answers or training admission.
