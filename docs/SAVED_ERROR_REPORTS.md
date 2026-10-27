# Saved synthetic error reports

These are ordinary pixel-only forwards on a saved synthetic model's original
probes. They decompose errors; they do not supply labels or geometry to the
model, change weights, decode a complete runtime state or establish real accuracy.
Run only after the training Job terminates and its process tree is verified gone.
Keep its complete/controlled-partial status and immutable source archive.

The worker verifies source/configuration/runtime/weights, reconstructs the exact
probe recipes and rasters, and reproduces every saved per-case metric before
writing format2. Each case retains those ordinary metrics with its board/layer/
geometry and optional component breakdown. M2's frozen old sampler receives no
new arguments. M3 uses its recorded final-capture policy and component capacity.

Queue diagnostics preserve absolute indices, separate pieces from EOS and mask
unknown identities/presence. Unknown middle entries do not shift the suffix;
unseen endpoints supply no queue length. Hold status and supported identity are
separate. Readable component denominators differ from matching only the known
fields. Results are raw heads, not a release policy or calibration result.

For the current9 training+18 development probes, budget27 forwards. Expected
peak is below600MiB from prior same-size M2 reports, experimentally capped for
M3; use1GiB/120s with native Job and OS pressure guards. Serialize model jobs.
Launch detached with hidden matching.log/.stderr.log and a checkpoint/PID entry:

```text
.venv/Scripts/python.exe scripts/run_bounded.py --memory-mib 1024 --wall-seconds 120 --work-items 27 --receipt runs/m2-stream-accuracy704-errors-r3-job.json -- .venv/Scripts/python.exe scripts/probe_stream_errors.py --run m2-stream-accuracy4096-s2 --output runs/m2-stream-accuracy704-errors-r3
```

The example replays the eligible saved704 artifact. s3 stopped at OS pressure
with bank1152 but no final inference artifact; do not analyze it as a completed
segment. Every output/receipt name must be new. Larger18..36
development populations require their actual finite count, not a copied27 cap.
The worker itself refuses more than45 total probes.

After verifying termination, curate under128MiB/30s with a finite work cap equal
to the report's actual probe count (expected collector peak<96MiB):

```text
.venv/Scripts/python.exe scripts/run_bounded.py --memory-mib 128 --wall-seconds 30 --work-items 27 --receipt runs/accuracy704-error-curation-r1.json -- .venv/Scripts/python.exe scripts/record_stream_errors.py --name m2-stream-accuracy704-errors-r3
```

The collector checks the native command/output/work/memory identity, actual
checkpoint/config/architecture, original partial tag, every sample/ancestry/
recipe, every ordinary metric and recomputed aggregates. It verifies unchanged
analysis sources against committed files, original archive and weight bytes,
then copies receipts, source/weight manifests, input summary, probe ledgers and
result with hashes into docs/evidence/<name>/. scope.json is written last.
Raw images/tensors remain in their original local run; no evidence is deleted.

Eight metadata fixtures use preserved704-report data, adapted in memory to the
format2 schema. They do no inference and do not upgrade historical reports.
With the27 existing decomposition/compatibility cases,35 tests pass in1.906s,
248,700,928B peak under512MiB/60s (runs/error-evidence-tests-r1). Native format2
M2/M3 replay and collector integration remain pending. Source-bound35-case scope:
docs/evidence/error-evidence-contracts/scope.json on77128c4 (32 selected hashes).
