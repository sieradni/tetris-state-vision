# Isolated type-head fit on frozen balanced features

The completed image models remain frozen. The saved feature observations prove
the exact existing type-head inputs and logits, but a weak centroid readout does
not establish linear inseparability. This experiment isolates the shared8-class
linear head, without another image-model training run or parser dependency.

Use both curated controlled banks separately: ordered and interleaved, each with
64 known training-role vectors from32 images/16 ancestries, eight labels per
class including absence. The18 development vectors never supply gradients,
normalization statistics, initialization choices or stopping decisions.

Predeclared recipe: copy each model's actual saved head; fresh AdamW optimizer,
learning rate.01, decay0, betas(.9,.999), epsilon1e-8. Full-batch cross entropy on
all64 known training vectors, fixed1,024 updates, no early stopping or tuning.
Use float64 internally with training-only per-channel centering/population
standard deviation, floored at1e-6, to condition the optimization. Transform the
initial head algebraically so its function is preserved. Reproduce original
float32 per-image logits exactly before optimization; compare conditioned versus
unconditioned float64 initial logits within1e-10. No model/label/source changes.

The final normalized head is converted back to a raw affine head. Report both
raw float64 and raw float32 predictions and conversion discrepancy. The finite
training-fit criterion is **64/64 correct training types from the raw float32
head**, including both roles/absence. Float64-only success is not this criterion.
This is type classification on saved vectors, not exact pose, locked boards,
complete state, converged architecture ranking or real-image generalization.
Failure at this finite budget is not proof of inseparability. A successful fit
would justify a separate pixel replay of the copied head before any model-quality
claim; it does not automatically change an inference artifact or select a model.

Admission: fresh per-bank warm characterization under640MiB/60s/work4. It takes
two warm updates and saves the head/Adam state; one continuous next update and
one reloaded next update must agree exactly in loss, gradients, parameters,
optimizer state and predictions. These extra two updates are explicitly charged
to work4. The published warm head remains at step2. A separate collector must
verify raw identities, metrics and native receipt before the full run.

The full run starts again from the original copied head, resetting only this
diagnostic optimizer. It must reproduce the warm first-two records exactly;
it does not resume or extend the original image study. Cap640MiB/120s/work1024,
inner wall100s,CPU1, no workers/prefetch; final tensor/optimizer reload exact.
Expected below500MiB based on the larger native feature observers; experimental
warm cap applies before admission. All source/input hashes checked before/after,
raw receipts/logs/startup/death evidence and separate tensor-only payloads are
preserved. Unknown labels are excluded; the declared balanced bank refuses if
its64 readable training labels are not available. Output reuse refuses.

Prior-defect interaction: no architecture/wire/scheduler/visibility change.
V-003/V-014/V-018/V-020..V-024/V-026/V-027/V-028 require artifact-derived identity,
frozen byte-verified sources, actual receipts and recomputed/readback aggregates;
bind actual feature/head/model bytes and use JSON-stable keys. V-001/V-002/V-005/
V-006/V-016 require image-supported labels: reuse the curated public targets,
never infer unknowns. V-004/V-011/V-012/V-017/V-019/V-023/V-025 require meaningful
validated instruments: balanced-class/ancestry, unknown/development exclusion,
initial-logit and optimizer-reload gates; retain pre-import CPU limits. Native
resource identity/pressure/cache/persistence controls remain unchanged.

Six pure input/label/ancestry/unknown/development/metric/JSON checks PASS in.023s
(native.265s/35,033,088B under128MiB/30s/work6),with exact source and raw native/
startup/death evidence in docs/evidence/type-head-contracts-r1/. This is not
neural/optimizer admission. The separately verified warm admissions and complete
fixed fits are recorded below.

Ordered warm admission PASS: native2.896s/333,045,760B under640MiB/60s/work4;
initial float32 per-image logits exact, conditioning discrepancy2.6645e-15,
continuous/reloaded next update,head/Adam states and predictions exact. Independent
collector PASS.381s/79,106,048B under128MiB/30s/work84,including recomputed
training-only mean/std,raw affine conversion and actual banked step2. Raw sources,
native/collector receipts/logs/startup/death evidence:
docs/evidence/type-head-ordered-profile-r1/. No full-fit verdict follows yet.

Ordered fixed1,024-update fit COMPLETE;native3.209s/332,963,840B under640MiB/120s,
first two optimizer records exactly match warm admission,tensor/Adam reload exact.
Independent collector PASS.392s/79,159,296B under128MiB/30s/work1106. All known
PIDs/commands gone;raw source/payload/native/collector/startup/death evidence:
docs/evidence/type-head-ordered-fit-r1/. Raw float32 types improve20/64->62/64
(active32/32,ghost30/32);the two remaining errors are T->S on both views of
ancestryall-class-04. The64/64 criterion FAILS;no extension is selected.
Development remains0/18;cross entropy worsens3.27902->58.59039. Float64/rawfloat32
class counts agree;raw affine conversion maximum discrepancy3.02e-14. This
establishes substantial class information in these saved vectors,not an image
model improvement or independent generalization. The interleaved result follows.

Interleaved warm admission also PASS: native2.877s/333,144,064B under640MiB/60s/
work4;initial float32 logits exact,conditioning error5.3291e-15,continuous/reloaded
next update and tensor/Adam state exact. Independentcollector PASS.349s/
78,077,952B under128MiB/30s/work84. Raw source/payload/receipts/logs/startup/death
evidence docs/evidence/type-head-interleaved-profile-r1/.

Interleaved fixed1,024-update fit COMPLETE: native3.254s/332,857,344B under640MiB/
120s,first-two warm records exact,tensor/Adam reload exact. Independentcollector
PASS.378s/79,425,536B under128MiB/30s/work1106. Raw sources,payload and native/
collector/startup/death evidence:docs/evidence/type-head-interleaved-fit-r1/.
Raw float32 types improve37/64->64/64,active32/32 andghost32/32. The declared
vector-fit criterion PASSES. Development remains0/18;cross entropy worsens
4.93344->25.48596. Raw affine conversion discrepancy2.84e-14;float64 and raw
float32 class counts agree. All known PIDs and matching commands are gone.

| Fixed feature diagnostic | Ordered | Interleaved |
|---|---:|---:|
| Initial training type labels |20/64|37/64|
| Final raw float32 training type labels |62/64|64/64|
| Final active / ghost types |active32/32;ghost30/32|active32/32;ghost32/32|
| Training cross entropy |1.60518->.15517|.77506->.05770|
| Development types |0/18->0/18|0/18->0/18|
| Development cross entropy |3.27902->58.59039|4.93344->25.48596|
| Declared64/64 vector-fit criterion |FAIL|PASS|

The interleaved frozen features support all64 balanced training labels with the
existing shared8x64+bias affine head in float32. The original type errors are
therefore not evidence that these particular vectors cannot represent the labels.
This does not identify why joint training left the original head underfit.
Both readouts overfit the small training bank;neither improves development.
They are separate conditioned full-batch optimizations,not an extension or
controlled convergence comparison of the original image-training recipes.

Next perform an explicit conditional pixel replay for each copied raw float32
head. Reproduce the original ordinary outputs and saved feature inputs first,
then change only the in-memory copy's type-head weights. Every non-type tensor
at every stage,original artifact/source bytes,and output head restoration must
remain exact. Report new type/pose/joint-fit metrics independently;do not apply
vector-fit counts to an image-model gate without executing it. Neither original
image artifact is changed or promoted. More optimization of these completed
head diagnostics is not selected.
