# Tetris State Vision

An independent image-to-state model research project. The first milestone
implements state/annotation contracts, honest exact-state scoring and synthetic
scene generation. **There is no trained model or screenshot parser yet.**

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

The first preview is [examples/montage.png](examples/montage.png); representative
complete and cropped states are in examples/. The state API is
`tetris_vision.state.VisionState`; strict conversion is
`tetris_vision.adapter.to_solver_input(state, hold_available=...)`.

The curated verification receipt is docs/evidence/foundation.json. No engine
runtime, detector, downloaded weights or existing user images are dependencies.

The design source is archived in docs/research/Tetris_Vision_Research.md.
See docs/ROADMAP.md for milestones and docs/ANNOTATION.md for label semantics.
