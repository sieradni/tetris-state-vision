# Contributor handoff

Start with [the core goals](docs/CORE_GOALS.md). This repository is independent
of `tetrguide`. No project jobs were left running at handoff, and the original
image models and completed experiments are preserved.

Repository: **https://github.com/sieradni/tetris-state-vision** (private).
The owner must grant your GitHub account collaborator access before you clone.
No collaborator has been invited automatically; their username was not supplied.

## GitHub and the ZIP have different jobs

**GitHub is the shared source of truth for code and history.** Clone it, make a
branch, commit your changes and submit a pull request. The ZIP is a transport
snapshot of one exact commit plus selected model/run artifacts that are ignored
by Git. Sending a ZIP does not create a Git remote or synchronize later changes.

Recommended workflow:

```powershell
git clone https://github.com/sieradni/tetris-state-vision.git
cd tetris-state-vision
git switch -c codex/conditional-type-head-replay
```

Restore the ZIP's extra artifacts into that checkout using the command below.
It checks that your checkout is at the ZIP's recorded commit, verifies every
archive hash, leaves tracked code untouched, and refuses to overwrite differing
local artifacts. If `main` has advanced, check out the recorded handoff commit on
your new branch before restoring; merge newer source changes afterward and rerun
the relevant gates rather than silently mixing provenance.

```powershell
python scripts/run_bounded.py --memory-mib 256 --wall-seconds 180 --work-items 12000 --receipt runs/handoff-restore-r1.json -- python scripts/handoff_bundle.py restore --zip C:/path/to/tetris-state-vision-handoff.zip
```

Launch that command hidden/detached with logs if it may exceed 30 seconds, as
shown for setup below. Receipt names must be new. The script's `verify` mode
checks a ZIP without restoring files. It has fixed file/byte budgets; it does
not delete old runs or replace an existing artifact with different bytes.

Alternatively extract the ZIP into a new folder to inspect a standalone snapshot.
It contains no `.git` directory, so that folder is not a clone and has no remote.
Use the clone-and-restore workflow for contributions. The archive's
`.handoff/manifest.json` records the source commit, repository URL, file hashes,
included artifact roots and exclusions. Verify the adjacent SHA256 file if the
archive was transferred through another service.

## What is included

Git contains source, tests, lockfile, docs, synthetic examples and curated research
evidence. The ZIP adds the three current image-model runs, their latest optimizer
snapshot per run, relevant warm admissions, both copied-head profiles/fits and
their current diagnostic reports/receipts. Frozen execution sources and saved
synthetic probe images are included so the next pixel replay has its inputs.

The archive does not contain `.venv`, credentials, Git's internal directory,
private screenshots/contact sheets, the neighboring engine, every historical
run or old optimizer generation. Excluded local data is preserved on the owner's
machine. Older reports can name artifacts outside this continuation pack; ask for
a specific older artifact if a new task actually requires it. Do not claim those
missing historical model files were freshly rerun or verified.

Historical receipts contain absolute paths from the original Windows machine.
Those strings are provenance; do not edit archived receipts to make them look
native to your computer. Current modules derive their root from the checkout.
If a legacy tool requires path adaptation, implement and verify a new explicit
adapter with fresh receipts, retaining the original bytes and source scopes.

## Environment and guarded execution

Reference execution is Windows, Python 3.14, `uv`, CPU PyTorch, one CPU-library
thread, no GPU. The project declares Python >=3.11; the named latest neural
receipts use the pinned environment described in their manifests and `uv.lock`.
The current job runner deliberately requires Windows kernel Job enforcement.
Another OS needs an equivalent enforced guard and admission before long compute.
The ZIP itself is a standard archive and can be inspected on any OS.

The delivery was checked by `scripts/verify_handoff_checkout.py`: it makes a fresh
GitHub clone, verifies/restores the archive, checks latest optimizer payload hashes
and runs one original pixel prediction per current model plus the saved copied-head
vector replays. The adjacent delivery verification report states the exact scope
and runtime. It reuses the owner's pinned runtime; it does not claim a fresh
dependency install or completion of the planned conditional pixel intervention.

Install Git, Python and uv. Run from the clone root. This example installs the
locked training environment without launching model training:

```powershell
New-Item -ItemType Directory -Force runs | Out-Null
$python = (Get-Command python).Source
$job = Start-Process -FilePath $python -WorkingDirectory (Get-Location).Path `
  -ArgumentList @('scripts/run_bounded.py','--memory-mib','1024',
    '--wall-seconds','600','--work-items','1','--receipt','runs/setup-train-r1.json',
    '--','uv','sync','--frozen','--extra','train') `
  -WindowStyle Hidden -RedirectStandardOutput runs/setup-train-r1.log `
  -RedirectStandardError runs/setup-train-r1.stderr.log -PassThru
Get-CimInstance Win32_Process -Filter "ProcessId=$($job.Id)" |
  Select-Object ProcessId,CommandLine
```

Before launching, state expected peak memory, the actual ceiling and the pressure
guard, and record the command/log/receipt/limits in the continuation checkpoint.
The example's install ceiling is 1 GiB; reduce it if your host's allowed clamp
requires it. Verify the exact command exists after launch and inspect the terminal
receipt/logs; absence of new terminal output is not success. The existing runner
records OS admission pressure, enforces aggregate commit/wall limits and sets CPU
library thread limits before import. Its generic `--work-items` declaration does
not magically bound arbitrary code: children must enforce their finite work.

After setup and artifact restore, a small meaningful starting check is:

```powershell
.venv/Scripts/python.exe scripts/run_bounded.py --memory-mib 128 --wall-seconds 30 --work-items 6 --receipt runs/contributor-head-contracts-r1.json -- .venv/Scripts/python.exe -m unittest discover -s docs/experiments -p test_type_head_fit.py -v
```

Use the same detached/logged launch pattern for tests and all longer work. The
six checks validate bank balance, pairing, ancestry, unknown labels, development
exclusion and metrics/JSON contracts; they are not a full-suite or neural-quality
gate. See `AGENTS.md` and [RESOURCE_GUARD.md](docs/RESOURCE_GUARD.md).

## First task and working agreement

The exact next acceptance-gated task is in [CORE_GOALS.md](docs/CORE_GOALS.md)
and [CONTINUATION_STATE.md](docs/CONTINUATION_STATE.md): conditional pixel replay
of the fitted raw float32 type heads, with original predictions and all non-type
tensors checked. Do not resume the finished studies merely to get more steps.

Keep changes narrow. Add meaningful checks for changed contracts, run them with
the appropriate limits, keep raw run data out of commits unless it is deliberately
curated evidence, and update the roadmap/checkpoint. Never broadly stage ignored
runs, screenshots or your virtual environment. Record a new defect with a named
green closure gate. Preserve existing hashes, frozen execution sources and partial
statuses. Do not raise ceilings, disable guards, rewrite shared history, publish
assets or delete another owner's data.

Push your branch and open a PR describing the problem, final behavior, evidence
and limits. No production deployment or public release is part of this handoff.
