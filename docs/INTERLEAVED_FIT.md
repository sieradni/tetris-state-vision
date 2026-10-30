# Completed order-only fit comparison

Both matched1,024-update studies are complete and independently audited. The
interleaved schedule improves visible piece labels on this fixed seed and32
controlled drawings, while exact boards and geometry regress. Both fail the
unchanged joint-fit criterion. No ordering is selected as a release solution,
and neither finished study is extended.

| Final diagnostic | Ordered | Interleaved | Required |
|---|---:|---:|---:|
| Exact locked boards | 32/32 | 28/32 | 32/32 |
| Exact visible piece type and pose | 12/56 | 30/56 | 56/56 |
| Exact absence | 8/8 | 7/8 | 8/8 |
| Correct supported semantic cells | 128/128 | 128/128 | 128/128 |
| Mean geometry error, cells | 0.211 | 0.386 | <0.25 |
| Exact nonempty development boards | 0/7 | 0/7 | Separate diagnostic |
| Exact development roles | 0/18 | 0/18 | Separate diagnostic |

All observed columns, rows and equivalent rotations are correct in both final
fit populations. The remaining role errors are type errors:44 ordered versus27
interleaved. Candidate exact visible labels by target class(total8 per class,
four views per role):I8,O4,T2,S4,Z2,J6,L4. This improves class coverage but leaves
substantial class-reading failure. Both exact development boards are empty;
the nine development views represent only three ancestries.

The final verifier confirms identical neural/loss/target package sources,
matching model/optimizer/input/step recipes and identical32-pair pixel/public
state populations in all32 completed cycles, with actually different training
order. The earlier matched-start proof supplies identical initial predictions
and same-input warm tensors. Fit verdicts are recomputed from each independent
ordinary raw-head report, not copied from a declared success flag.
Evidence: docs/evidence/interleave1024-comparison-r1/scope.json; native metadata
comparison.350s/51,212,288B under128MiB/30s. Candidate final ordinary41 replay
7.370s/424,361,984B under640MiB and independent collector.398s/78,548,992B under
128MiB. Actual final weights93e798208234eb7d54ace1b725949b0cab848c7d0a464520d8c23a6dd88a0273,
catalogue68bb40e3. Source/controller/snapshot/native receipts are retained in
m2-stream-interleave1024-s3; raw errors in m2-stream-interleave1024-errors-r1.

Next complete the existing larger4,096-update baseline from its verified2,816
checkpoint with the original frozen trainer/optimizer. Retain both failed fit
studies; inspect the class-feature path before adopting another mechanism.
This comparison does not establish independent-generalization or real-image
quality. The locked fit population contains only empty boards and one fixed
eight-cell bottom row, not a broad board-reading benchmark.

## Protocol and earlier checkpoints

This follow-up is selected after the original1,024-update study and its independent
ordinary replay failed the fit gate. All visible positions/rotations fit, but44
piece labels are wrong; predictions use only Z/J/L and absence. See ALL_CLASS_FIT.
The new policy is integrated with88 selected checks on25251c8. A later profile
ancestry fix has its own8-case/source-bound and native replay gate (V-027).
Fresh warm admission, same-input numerical comparison and exact continuous/
recovered training pass on catalogue68bb40e3. The completed studies retain their
declared sources,schedules and targets. The protocol below records the planned
controls and earlier partial checkpoint evidence.

Both studies' initial predictions/losses and model/optimizer recipe match
exactly. The first four completed32-step cycles contain identical pixel/public
state pairs once each, with a verified different optimizer sample order.
Evidence: docs/evidence/interleave128-matched-start-r1/scope.json. The warm
same-input comparison also has byte-identical actual weights and unchanged
neural/loss/target package sources. This supports the comparison's controls;
it does not decide whether interleaving improves learning.

| At128 updates | Ordered | Interleaved |
|---|---:|---:|
| Exact locked boards | 28/32 | 32/32 |
| Exact visible piece type and pose | 0/56 | 2/56 |
| Exact absence | 8/8 | 4/8 |
| Correct supported semantic cells | 128/128 | 128/128 |
| Mean geometry error, cells | 0.216 | 0.303 |

Both partial checkpoints fail the unchanged joint-fit criterion. Neither has
correct nonempty development boards or development roles; their two exact
development boards are empty. Final1,024-update evidence remains required.
Candidate ordinary replay7.460s/425,586,688B/640MiB and independent collector
.396s/78,340,096B/128MiB reproduce every saved score; named scope is
docs/evidence/m2-stream-interleave128-errors-r1/scope.json.

Question: does interleaving classes improve fit when the images, public labels,
model, initialization and optimizer are unchanged? The original cycle contains
four consecutive views per class and always ends with four absences. Its640
checkpoint predicts L on26/32 views per role; observed successes are all L.
Recency is a hypothesis, not an established explanation or a code defect.

The candidate policy is all-class-interleaved-v1, fixed order seed20261027.
Each32-update cycle contains exactly the same32(index,view) pairs once. Four
eight-update rounds each contain every7-class/absence label once. SHA256 ranks
keyed by policy/seed/global cycle independently order classes per round and the
four pose/style pairs per class. This consumes no Python/Torch RNG and does not
depend on evaluation time or optimizer outcomes. Recover order from global step,
not a mutable shuffle cursor. Every cycle is bounded; at most4,096 global steps.
Ancestry counts use the actual prefix; full cycles reuse16 groups, not diversity.

No rendering, crop, prepared target or observation policy changes. Require exact
equality of all32 original pixel/state/recipe identities under the permutation,
plus every-cycle pair/class/pose/style coverage and arbitrary-cursor recovery.
Fixed probes keep original index/view order. Keep stream-v1 and all-class-fit-v1
numerically and behaviorally unchanged; the new policy is M2/capacity0 only.

Integration contract: include the new module in the frozen archive and explicit policy
identity. Require fresh source-bound protocol checks, full-square warm profile,
matching resource admission, actual continuous/recovered training and curation.
Old checkpoints/admissions cannot be relabeled as the new policy. Compare fresh
initialization seed20261026,pilot/layeroff,AdamW LR.001/decay.01,CPU1/cache16,
eval128/snapshot64,dev3groups/9views, exactly1,024 updates (128+512+384 segments).
Keep the same32/32 boards,64/64 supported roles,128/128 semantic cells and<.25
geometry criterion. Bind initial tensor/prediction identity to the original fit.
Inspect a final independent ordinary raw-head report. Failure remains failure;
no extra steps or model changes may be silently included in this comparison.

Prior-bug interaction review: the current BUG_LOG was read before this plan.
V-001/002/005/006/016: reuse observer-supported raster/target identities; no
new visibility policy or latent labels. V-003/014/018/020/021/022/024/026: bind
actual source/policy/global order/recipe/artifact identity and raw file bytes;
freeze live controllers/helpers and independently recompute aggregates.
V-004/011/012/017/019/023/025: native measured admissions, supported fixtures,
warm optimizer, pre-import thread limits, explicit normalized/raw-copy rules.
V-007/010/013/015: unchanged named kernel Job, OS commit pressure guard, finite
RAM/wall/work, bounded cache and atomic snapshots. V-008/009: real draft corpus
remains ineligible. This is a sampling comparison, not an architecture adoption.

Standalone sampler checks: seven pass in28.595s(native29.540s/41,766,912B under128MiB/30s). This covers all128 cycles, global mapping/recovery boundaries, strict finite inputs, no Python RNG use, actual prefix ancestry and96 public pixel/state/recipe replays. It is not a neural training or admission gate.
