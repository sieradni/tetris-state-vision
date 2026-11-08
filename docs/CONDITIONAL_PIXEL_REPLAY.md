# Conditional copied-head pixel replay

The fixed ordered and interleaved 1,024-update image models and their copied
head fits remain frozen. Each intervention ran the 41 saved synthetic images
twice, first with the original head and then with the fitted raw float32 head
in memory. Inputs were pixels and validity only. The original ordinary reports,
final head inputs and logits matched the independently curated observations.
All non-type stage and final output tensor bytes and all type-head stage inputs
matched between the two passes; the original head was restored after each case.
The model state, curated input identities and execution sources were checked
again after replay. No original checkpoint or optimizer was written.

| Training diagnostic | Ordered | Interleaved |
|---|---:|---:|
| Exact locked boards | 32/32 | 28/32 |
| Exact visible active/ghost labels | 54/56 | 56/56 |
| Exact absence labels | 8/8 | 8/8 |
| Correct supported semantic cells | 128/128 | 128/128 |
| Mean geometry error, cells | 0.21059 | 0.38618 |
| Declared joint-fit criterion | FAIL | FAIL |

The joint criterion requires 32/32 boards, 64/64 known role labels, 128/128
semantic cells and mean geometry error below 0.25 cells. The ordered model fails
on two type labels; the interleaved model fails on boards and geometry. Both
models still score 0/18 development type labels. The changed pixel logits
matched the saved raw-float32 vector logits exactly in these executions; this
observed zero difference is not a general guarantee about floating-point matrix
operations. Thirty-two training images share 16 ancestries, and the nine
development images share three. This is a conditional intervention on saved
synthetic fixtures, not a trained replacement model or evidence of real-image
accuracy or generalization.

Independent collectors recomputed the metrics from packed pixel predictions and
public targets, checked the original decomposition against the saved ordinary
reports, audited native receipts and source/input identities, and checked
pixel/vector class counts. Raw predictions, targets, receipts, logs, source
copies, derived architecture/weight identity and detailed metrics are preserved
in [ordered evidence](evidence/conditional-pixel-ordered-r1/scope.json) and
[interleaved evidence](evidence/conditional-pixel-interleaved-r1/scope.json).
Native peaks were 426,795,008 and 426,926,080 bytes under 640 MiB / 120 s /
82-forward limits; independent collectors used 41,111,552 and 41,107,456 bytes
under 256 MiB / 30 s / 41-case limits. All corresponding processes terminated.

The result supports investigating joint optimization of the type head: features
from both fixed image models support substantially better training-label fits,
while the original jointly trained heads underfit those same features. It does
not establish a cause or solve the severe development failure. Before training,
predeclare a controlled joint-training comparison and resource/recovery gates;
retain board, pose, type and geometry criteria and evaluate independent ancestry
coverage. Do not extend the completed studies or promote the intervention.
