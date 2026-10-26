# Tetris State Vision

An independent image-to-state model research project. It includes state contracts,
exact-state scoring, synthetic scenes and the scratch M2 lattice-query model.
**There is no released complete screenshot parser or validated model yet.**

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

The inspection API is `ImageParser.from_checkpoint(path).parse_image(image)` in
`tetris_vision.runtime`. It uses the full image, always returns needs_review (or
unsupported), and leaves queue/hold unknown until M3. Run the guarded CLI with
scripts/inspect_m2_checkpoint.py; its real 16-step-artifact call and seven focused
contracts are scoped in docs/evidence/m2-runtime/. The resulting JSON is there
as state.json. This path is for inspection, not automatic solver input.

The design source is archived in docs/research/Tetris_Vision_Research.md.
See docs/ROADMAP.md for milestones and docs/ANNOTATION.md for label semantics.

Local real-image preparation and audit are described in [docs/CORPUS.md](docs/CORPUS.md).
Imports stay drafts until independently reviewed and permitted. Hashes, related
image groups and externally pinned held-out snapshots protect benchmark identity;
these structural checks are not evidence of label correctness or model accuracy.
