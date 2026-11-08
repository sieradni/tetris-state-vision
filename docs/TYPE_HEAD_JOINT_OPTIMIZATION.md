# Controlled joint optimization of the type head

## Question and fixed comparison

Can a higher learning rate for the existing shared type head reduce the
observed training-type collapse without losing locked-board and geometry fit?
The copied-head pixel replay shows that frozen training features support a
better affine readout; it does not show that learning rate caused the collapse
or that this change improves development. This is a diagnostic, not a model
selection or release study.

Compare a fresh same-source control with a candidate. Both start from seed
20261026 and use the ordered all-class-fit-v1 32-image population, pilot M2,
capacity zero, no dense layer evidence, 1,024 updates, batch one, AdamW base
learning rate .001 and weight decay .01, clip norm 1, evaluation every 128,
snapshots every 64 and three development ancestries. The candidate changes only
the learning rate for `layer_type.weight` and `layer_type.bias` to .004; all
other tensors remain at .001. Fourfold is a fixed diagnostic contrast, not a
tuned optimum. Default factor one must preserve the existing single-group
optimizer and numerical behavior. No fitted head initializes either run.

Predeclare success as 32/32 exact boards, 64/64 exact known roles (including
eight absences), 128/128 supported semantic cells and geometry below .25 cells
on training probes, with no qualification from loss alone. Report all type,
pose, active/ghost and absence errors, board false positives/misses, geometry,
and development metrics for both models. Keep the fixed criterion even if
the candidate improves only some scores. Compare every actual recipe and data
prefix, initial weights and outputs, total update budget, frozen source and
optimizer identity. Development is descriptive, with nine views of three
ancestries, and neither this memorization test nor a pass establishes real
screenshot quality or generalization. Do not tune on development or extend
either completed 1,024-update experiment.

## Admission before any training study

First implement the opt-in rate and bind it to source, resource admission,
optimizer parameter groups, recovery contract and controller routing; reject
factor/profile/recovery mismatches. Tests must cover factor one equivalence,
exactly the two head parameters in the candidate group, invalid values,
admission refusal and recovery/route identity, and preserve existing runner
environment safeguards. Obtain a fresh full-square two-step warm/optimizer
and reload profile for each factor under 640 MiB and verify exact recovery on
the changed recipe before any fit. Demonstrate identical model initialization,
input recipes and initial outputs and that only the selected group rate differs.
Run a continuous versus interrupted/resumed matched-state recovery gate for
the candidate and verify model, optimizer, RNG, ledgers and probes.

After these gates, perform finite separately bounded segments with existing
pressure guards, logs, receipts and checkpoints. Recheck more than 40 GB free
before each long training window. Use at most one heavy job at a time; do not
relax the machine limits. Independently replay all final saved pixel probes and
derive both joint-fit verdicts from actual outputs. The earlier models, original
baseline, their artifacts and historical evidence scopes stay unchanged.

## Admission progress

The opt-in factor and controller/recovery routing have focused tests. Both
factors completed separate full-square warm profiles and native continuous6
versus partial3+resume3 gates, with independent recipe and pixel replay:
[candidate r7](evidence/m2-stream-recovery-gate-r7/scope.json) and
[control r8](evidence/m2-stream-recovery-gate-r8/scope.json). The shared frozen
catalogue is `c69372af51acb182ca5a883894afe75081d152be39e691694b8fd8e623b3b70b`.
Peaks were 602,124,288 and 600,768,512 bytes under 1 GiB parent limits; each
child was limited to 640 MiB. Independent collectors replayed 17 training
records and 164 saved probes per gate under 128 MiB.
The collectors' `source_base_commit` names the parent commit at collection;
the then-uncommitted training sources are identified by the frozen catalogue,
and were byte-checked against the sources in this milestone before committing.

The bounded matched-admission comparison checks actual saved optimizer groups
and learning rates, identical initial probe metrics and first loss/gradient norm,
identical six training recipes, and identical non-head weights after the first
profile update. Candidate head weights differ after that update. Exact initial
weight bytes were not saved: equal initial metrics and a matching seed do not
independently prove byte identity. Thus the predeclared initial-weight check
remains open and the full fit is not yet admitted. Capture and verify initial
tensors in a fresh source-bound comparison, then repeat affected profiles and
recovery gates. No model-quality claim follows. Receipts and comparison:
[admission evidence](evidence/type-head-admission-contracts-r1/).
