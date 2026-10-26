# Workload memory and time guard

`scripts/run_bounded.py` keeps the original readback-verified Windows aggregate
Job commit ceiling and kill-on-close, finite wall budget and exclusive durable
receipt. The child still enforces the declared work count. No unlimited-memory
option exists. Limits cannot exceed min(physical RAM / 3, 8 GiB).

V-010 adds a system-pressure candidate. It uses Windows
[GetPerformanceInfo](https://learn.microsoft.com/en-us/windows/win32/api/psapi/nf-psapi-getperformanceinfo)
and its [page counters](https://learn.microsoft.com/en-us/windows/win32/api/psapi/ns-psapi-performance_information),
converted to bytes. These describe system commit and physical memory, rather
than an incidental nested Job's counters. Current system commit is distinct from
the historical peak Job accounting discussed in V-004.

The current conservative policy stops at the lower of:

- Physical memory minus min(2 GiB, physical / 8).
- 90% of the current OS commit limit.

It also requires at least min(2 GiB, physical / 8) available physical memory.
Admission checks headroom for the full configured Job ceiling. While the child
runs, system counters are polled every .2 seconds, independent of snapshot/log
cadence. A pressure stop writes `memory_pressure` / exit125 before terminating
the complete owned Job. Counter-query failures fail closed. There is no env or
CLI opt-out and no automatic restart. It never kills another owner's processes.

The supervisor records/exports its source hash and guard identity. The streaming
worker and longer-run admission require those identities to match the archived
runner and resource probe. An older supervisor or pre-guard resource receipt
cannot admit a current longer study.

**Verification is partial; V-010 remains OPEN.** Before-change synthetic-pressure
admission failed as expected: the child started. The new real OS reader refused
a launch at 32,116,695,040 B commit against a 31,368,351,744 B stop point, with
the native cap armed and no child created. The parked test harness did not run
its five tests. Evidence is in `docs/evidence/v010-admission-only/scope.json`.

Still required: tiny-Job synthetic pressure and counter-failure process-tree
death tests, global-counter/nested-Job test, existing native allocation/wall
regressions and strengthened stream-admission tests. No host-scale allocation
is used by those fixtures. Run them when actual system headroom admits the
bounded harness; do not bypass the guard to obtain a green result.

The old streaming square characterization remains historical evidence of its
exact source catalogue. After guard verification, use a fresh output/receipt to
re-profile the changed catalogue before longer learning. A pressure park is not
a crash, completed learning run or accuracy gate. Preserve its receipt/logs.
