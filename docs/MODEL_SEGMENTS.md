# Finite original/joint model segments

`run_stream_segment.py` supports the original M2 and optional joint M3 model.
It derives model kind, configuration, component capacity and child memory ceiling
from the actual verified resource profile and checkpoint manifest. M2 names use
`m2-stream-...`; M3 names use `m3-stream-...`. An optional
`--component-capacity 0|10..64` is an assertion, not a configuration override.

Both fresh studies and resumes execute an archived trainer. A fresh study uses
the admission's archive; a resume uses its parent's archive after checking the
same source/configuration against admission. Current mutable package code cannot
silently replace an old study's math. Old M2 trainers receive no new component
keyword. Joint trainers use their recorded capacity/final-capture policy.

Resume inherits seed, learning rate, evaluation cadence, checkpoint cadence and
development groups from the committed snapshot. Source/config/global target,
input policy, CPU/cache recipe and finite parameter bounds must match. A finished
diagnostic cannot silently become a longer study. Budget is at most512 new
updates, clipped to the remaining original global target. Unknown component
labels, independent board/active/ghost targets and bottom-first rows are untouched.

Preflight happens before any output directory or child is created. A guarded
`--plan-only` verifies the actual files and prints the exact plan without starting
training. Example, under128MiB/30s/work1 with fresh logs/receipt:

```text
.venv/Scripts/python.exe scripts/run_bounded.py --memory-mib 128 --wall-seconds 30 --work-items 1 --receipt runs/m2-s4-preflight-r1-job.json -- .venv/Scripts/python.exe scripts/run_stream_segment.py --name m2-stream-accuracy4096-s4 --target-steps 4096 --updates 512 --admission runs/m2-stream-accuracy-probe --resume-from runs/m2-stream-accuracy4096-s3 --plan-only
```

Expected preflight RAM<96MiB; it does not load Torch or update a model. Inspect
the printed source/cursor/worker command and actual current host memory/disk,
then remove `--plan-only` for the separately logged bounded training pipeline.
Never use the small preflight Job as the training parent.

Training parent:1280MiB/2100s and at most512 new updates. Serialized training
child uses the exact admitted640MiB or1024MiB ceiling,1860s outer/1800s inner;
metadata replay uses128MiB/120s. Kernel named Job, OS pressure guard, pre-import
one-thread policy and16-entry native cache remain unchanged. No automatic retry
or guard opt-out. Expected total RAM<1100MiB for the measured larger envelope.
Launch detached with hidden logs/state/checkpoint and verify exact PIDs.

Controlled partial/complete cursors, actual artifact/source/configuration,
resource work counts and unique probe population are checked before curation.
Replay work is global training rows plus the actual9+3*development_groups probes.
Result architecture/capacity comes from the verified artifact. The controller
captures its source identity before planning and refuses a verified result if
the file changes while children run. Do not edit an active controller file.

Sixteen scoped tests pass in1.298s,57,577,472B peak under192MiB/60s. Byte-level
source validation is isolated in planning fixtures and has separate source gates;
dummy fixture trainers never execute. The negative named-Job check uses a real
subprocess. These tests do not establish actual M3 training, native end-to-end
controller recovery, long varied-shape memory or model quality. New actual
1152 preflight/continuation and joint controller integration remain pending.
