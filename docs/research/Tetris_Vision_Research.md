# Standalone Tetris image-to-state model
## Task-specific architecture, synthetic training, and research decisions

Revised research: 2026-10-25 (America/Los_Angeles). This revision supersedes the initial detector-led recommendation. It specifies a standalone custom model as the main research track. Existing parsers are comparison baselines. No new model has been trained or benchmarked.

Repository context: a2d5391 with pre-existing engine modifications left untouched. No runtime code, training job or search process was launched for this research. The engine roadmap remains separate.

## 1. Decision

Pursue a single, task-specific, jointly trained image-to-state model. It should learn the board's coordinate system and game-state components directly from the image, including the absence or partial visibility of components. It should have no inference dependency on the current detector, color thresholds, skin templates, or a succession of independently trained vision tools.

A promising design is a high-resolution image encoder with a board-lattice decoder, cell queries, active/ghost heads and a linked queue/hold decoder. Train a fully random-initialized version as a first-class candidate. Compare it with compatible pretrained initialization and a simpler direct decoder. Let whole-state real-image performance determine the winner.

The current parser's low accuracy and imprecise boxes, as reported by the user, are the problem to solve; existing region-detection scores are not an acceptance criterion. The new design is free to replace the vision implementation. Shared file formats, the solver adapter, basic tensor operations and export libraries do not constrain the learned perception architecture.

Three distinct choices must not be conflated:

| Choice | Meaning |
|---|---|
| Custom architecture | Choose image representation, coordinate learning and output heads for this task. |
| Training from scratch | Start the selected trainable parameters at random rather than external pretrained weights. |
| Standalone deployment | Ship one image-to-state tool with a clear API and owned model artifacts, irrespective of weight initialization. |

A custom architecture does not require randomly initialized weights to remain custom. Conversely, wrapping a generic detector in new application code does not make its perception task-specific.

My recommendation is to build the custom architecture and give scratch training a serious, adequately converged evaluation. Evidence does not yet establish that scratch weights, a large pretrained encoder or one particular decoder will be best. Unlimited synthetic labels make scratch training credible; they do not make visual coverage automatic.

## 2. What the current research landscape contributes

The closest useful analogy is structured visual recognition: recover the coordinate system, local content and relationships together. It is closer to recognizing a table or board configuration than to detecting a few generic objects.

| Primary source | What the reviewed work actually establishes | Design implication and limit |
|---|---|---|
| TABLET, ICDAR 2025 [R1] | Represents row/column splitting as directional sequence labeling and merging as grid classification. Its split and merge models are trained separately. | Grid-centered representation can avoid unstable box predictions. Borrow the representation idea; do not misdescribe TABLET itself as one jointly trained network. |
| FastTab, May 2026 preprint [R2] | Combines an FCN encoder, a global recursive module, axial transformers and grid-conditioned cell reading. The formulation assumes a single table or cropped table region. | Row/column reasoning and compact iterative refinement are relevant. Full arbitrary screenshots and Tetris semantics need additional design. |
| ConRTF, July 2026, accepted ICDAR 2026 [R3] | Uses task-specific boundary supervision to address the fact that small localization errors can corrupt cell assignment. It retains a detector-based framework. | Precision should be trained against structural consequences. It does not prove that all detectors are inherently imprecise. |
| TableSeq, April 2026 [R4] | Jointly generates structure, content and coordinates from an image through one sequence interface. | A direct image-to-state sequence is a credible competing architecture. Tetris has much simpler output than arbitrary tables. |
| Spatial Transformer Networks [R5] | Introduces differentiable image/feature transformations inside a learned network. | Lattice sampling can be part of the model, receiving gradients from recognition. |
| Deformable DETR [R6] | Uses sparse learned sampling around reference points to improve spatial attention and small-object handling. | Cell queries can inspect multiple informative locations instead of one color sample. |
| D-FINE [R7] | Refines localization distributions across decoder layers. | Coordinate uncertainty and bounded iterative refinement are worth testing. General AP is not a grid-precision measurement. |
| DEIMv2, January 2026 revision [R8] | Adapts DINOv3 features with a spatial tuning adapter; includes other lightweight encoder variants. | Global semantics and fine detail can need different treatment. This is an architecture reference, not a required dependency. |
| Rethinking ImageNet Pre-training [R9] | Shows randomly initialized detection models can reach competitive target-task accuracy with sufficient training; pretraining can accelerate early convergence without determining final accuracy. | A short scratch trial against a converged pretrained model is an unfair comparison. Results on COCO do not predict this task. |
| TADA, March 2026 revision [R10] | Studies targeted augmentation of difficult training examples rather than uniform expansion of all data. | Use failure-directed generation, while retaining a representative distribution. Its diffusion procedure need not be adopted. |
| DINOv3 / SAM 3 [R11-R12] | Provide general dense features / promptable segmentation respectively. | Useful initialization, annotation or comparison candidates; neither outputs this game's formal state as supplied. |

These sources support architecture mechanisms, not a published 95% exact-state Tetris result. Table scores such as TEDS or GriTS are not interchangeable with the percentage of completely correct game states.

Existing Tetris tools remain useful comparisons. snip2fumen documents standard-color/visible-grid assumptions. The public Tetromino Detection owner page reviewed earlier lists 93 images and detection classes, not complete cell-state annotations. Sketris's author describes image parsing, but the accessible public material did not establish broad exact-state performance. [R20-R22]

This review did not locate a validated drop-in model meeting the complete stated requirements. That is a scoped finding, not a proof of nonexistence.

## 3. Proposed model: TetrisStateNet

TetrisStateNet is a working name for this proposed architecture, not a published or validated model. It is one trainable model with internal structure, one public image input, and one versioned state output.

### 3.1 Shared image encoder with two spatial scales

Keep an image's aspect ratio, source scale and valid-pixel mask. Avoid collapsing a large screenshot to 640 by 640 before its smallest useful minos can be examined.

A concrete first architecture configuration to implement and measure:

- A convolutional detail path with features at strides 2 and 4.
- Context features at strides 8 and 16, using residual convolutional blocks and windowed attention.
- Stage widths around 48, 96, 192 and 256 channels; fuse selected scales into 192-dimensional decoder features.
- A small global-context branch for board instances, panel association and transition state.
- Six decoder blocks as an initial experiment, with deep supervision and two or three coordinate-refinement updates.

These are starting settings, not measured parameter counts or optimal sizes. Implement an accuracy-focused variant and a smaller variant after measuring actual capacity/compute. Preserve detail before reducing model size.

For very large screenshots, test internal learned sampling of source-resolution features. An internal differentiable glimpse is still part of a single model; it does not require an externally guessed crop. Always retain a global context path, so local sampling can recover from an inaccurate initial hypothesis.

If the source itself contains only a few ambiguous pixels per mino, resizing or learned super-resolution does not create reliable evidence. Minimum recoverable source scale belongs in evaluation.

### 3.2 Board-instance and lattice decoder

Use learned board-instance queries to find candidate game boards and associate their components. A fixed board-slot budget is an implementation capacity that must be reported when exceeded, not a universal scene guarantee.

For each candidate, predict a continuous lattice. For ordinary screenshots an affine map is enough:

p(c,r) = o + (c + 0.5)u + (r + 0.5)v

Here o is the projected bottom-left board corner, u and v are the image-space column and row basis vectors, and c/r are canonical column and bottom-first row indices. A projective extension can be tested for photos/skewed captures. Do not begin with an unconstrained arbitrary warp that allows the model to fold cells or hide alignment errors.

Supervise the lattice at cell centers, row/column boundaries and known visible anchors. Normalize errors by source cell spacing. Training and evaluation should care whether a cell is sampled correctly, not simply whether a large rectangle overlaps the board.

Visible grid lines are optional visual evidence. The lattice exists because cells have a spatial arrangement; it is not a Hough-line detector or a requirement for drawn separators. Use synthetic examples with no grid, no border and transparent board background from the start.

When a crop removes the bottom edge or enough structure to make row indexing ambiguous, preserve that ambiguity. Do not extrapolate a confident complete board merely because the output schema expects twenty rows.

A discrete set of geometry hypotheses or coordinate distributions can be compared with point regression. Averaging two plausible grid alignments may create an invalid half-cell alignment; ambiguity should survive long enough for the state reader to resolve it.

### 3.3 Cell-query decoder

Instantiate a query for each canonical cell in the supported visible board and a configurable spawn margin. For a common 10 by 20 board plus six rows above, that is 260 cell queries per board.

Each query receives:

- Its row/column identity.
- Features sampled from several subregions around its predicted cell.
- Access to coarser global features.
- Row/column attention and a smaller global-context interaction.

Attend to cell interior, outline and neighboring cells instead of assuming a mino has a bright center. Restrict learned sampling offsets relative to the lattice so queries cannot drift into a different logical cell unnoticed.

Outputs include locked occupancy, observability and local layer evidence. Keep an auxiliary per-pixel or fine-grid mask objective to retain information about thin outlines, translucency and edge fragments.

The locked board is predicted directly. It is never produced by counting all visible minos and deleting the cells of a guessed active piece. This specifically avoids the destructive operation in B-260.

Row/column attention supplies context without imposing false gameplay rules. Arbitrary puzzle boards can contain floating blocks, and clears/garbage mean occupied cells need not partition into tetrominoes or have a count divisible by four.

### 3.4 Active and ghost heads

Use a board-level active query that predicts presence, type and canonical footprint/pose. Tie its four-cell structure to tetromino geometry, but permit clipped pieces and an uncertain/no-active transition state.

Add a ghost head and fine image-layer evidence. Active, ghost and locked semantics are not reducible to an intensity ordering. A custom ghost can be brighter than the stack; an active piece can touch identical-colored blocks; shadows can occupy empty cells.

Jointly train footprint consistency, cell evidence and phase. Preserve sub-cell visual position separately when falling animation is not snapped to the cell grid.

The active piece may descend beside a taller tower. “Highest object” is not a universal active-piece test. A perfectly legal tetromino silhouette can also be part of locked stack.

When pre-lock and post-lock pixels are identical, single-frame certainty is impossible. The model can expose alternative state hypotheses or request one unresolved fact; it must not be taught to invent hidden timing state.

### 3.5 Queue and hold decoder

Use component-role queries sharing the image encoder and board-instance features. Learn whether an icon belongs to hold, a future queue position, another player or decoration.

A queue item can have auxiliary 4 by 4 shape supervision along with its piece identity. Shape targets discourage fixed color lookup while retaining legitimate visual evidence. The model should learn disabled/grayscale hold appearances and independent preview/board skins.

For variable length, compare a set-of-icons decoder with an ordered index-conditioned decoder. The preferred experiment is a shared next-k decoder evaluated in chunks with end/presence predictions. Reuse its parameters at additional indices rather than allocating a different classifier for each queue length.

The model learns icon position, next-order relationships and whether the visible end is established. It supports horizontal, vertical and wrapped arrangements without a hard-coded screen direction. Test length generalization beyond the training range, repeated adjacent pieces, and unknown middle slots.

No finite deployment has unlimited capacity. Limit image work and maximum generated items explicitly and return truncation if exhausted. A cropped panel's visible endpoint is not the end of the game's hidden queue.

Hold outputs distinguish a piece, confirmed empty panel, absent panel and unreadable/cropped panel. Queue absence similarly has an explicit status; zero recognized icons is not automatically a known empty queue.

### 3.6 Joint refinement and output

The cell/active/queue/hold branches exchange board-conditioned context. Recognition losses propagate through differentiable sampling into lattice learning. Intermediate decoder outputs receive supervision so localization and semantics can improve together.

Keep refinement bounded. Compare one update against two and three; retain extra iterations only if they improve exact-state performance. An unbounded test-time optimizer or external solver is not the default parser.

Final deterministic code serializes and checks the result. It must not silently rewrite cells because another board looks more playable. Parsing identifies image-supported state; it does not solve an inverse puzzle by substituting a plausible one.

This topology is custom to the task, assembled from established neural building blocks. A claim of scientific novelty would require a separate literature/experimental argument. Custom suitability does not require a new attention operator.

## 4. Alternatives that deserve controlled experiments

### Direct image-to-sequence model

Serialize board rows, active, queue, hold and statuses into a constrained sequence with explicit delimiters. A shared encoder and autoregressive decoder can produce the complete state directly. Pix2Seq and TableSeq supply relevant precedents for generating structured labels and coordinates from pixels. [R4, R13]

This approach has a compact public interface and naturally variable queue length. Potential weaknesses are serial latency, token omissions and propagation of early mistakes. Those are testable risks, not evidence that it must lose.

Use grammar-constrained decoding and explicit row width, so syntax is guaranteed. Syntax correctness still does not establish correct cells. Compare full-state exact match and real latency against the lattice-query model.

### Dense coordinate-field model

Predict board-instance membership plus canonical row/column coordinates over visible image pixels, then pool/splat evidence into logical cells. This may handle irregular visual fragments without relying on a center sample.

It introduces harder instance association, coordinate collisions and unobservable-background targets. Keep it as an escalation if rigid-lattice errors dominate on the declared inputs. A cell-query model is the more direct first specification for regular digital boards.

### Pretrained encoder with the same custom decoder

DINOv3 can initialize an alternative encoder while keeping the proposed state representation. A fine-detail adapter may be necessary because patch features alone can lose the information this task needs. DEIMv2 is relevant evidence that semantic features and fine localization detail can benefit from different paths. [R8, R11]

Run scratch and pretrained versions of the same compatible encoder/decoder to isolate initialization. Compare the custom convolutional hybrid separately. Otherwise architecture and pretraining effects are confounded.

## 5. Synthetic scenes: rendering, compositing, then capture damage

The user's suggested backgrounds, transparency, missing grid, crops, shadows and missing panels should be core training dimensions. Some belong in the scene renderer, while others belong in postprocessing.

### 5.1 Layered scene renderer

Build a render graph with independent layers:

1. Arbitrary outer background.
2. Board backdrop, potentially transparent.
3. Optional grid and borders.
4. Locked minos and garbage.
5. Ghost and shadows.
6. Active piece and motion trail.
7. Queue/hold icons and their panels.
8. Effects, text and foreground overlays.

Actual layer order, blending and effects vary between renderer families. Preserve that variation rather than assuming one order is universal.

Alpha compositing for a layer is:

I_next = alpha * foreground + (1 - alpha) * I_previous

Vary alpha per layer, per skin region and where appropriate per mino, not merely by fading the whole screenshot. Handle premultiplied alpha correctly. Match real renderer behavior for sRGB versus linear blending, additive glow, outlines, shadows and blur.

Backgrounds should include photographs, gradients, textures, games, video-like scenes, text, block patterns and counterexamples resembling minos. A transparent board over a new background changes the pixel evidence inside every empty and semi-transparent cell. Replacing only pixels outside an opaque board does not train that condition.

Generate flat, beveled, outline, hollow-center, textured, disconnected-looking and emissive skins. Board/active/ghost/preview skins can differ. Ensure logical occupancy is recorded independently from a skin's visible pixels.

No-grid examples should be present throughout training. Vary line width, spacing appearance, opacity, border-only views and low-contrast grid. The model should not associate missing grids with one particular client, color or background type.

### 5.2 True component absence versus cropped visibility

Randomly omit hold, queue, ghost or border in the scene itself. Separately crop them partially or completely after rendering. These conditions need different labels.

Store both latent scene state and what the final image supports. An unseen hold piece in renderer memory is not available to the parser. A partially obscured queue item remains a slot with uncertain identity; it must not disappear from the ordered list.

Randomize layout, icon scale, panel position, gaps and wrapping independently enough to prevent a fixed-layout classifier. Include multiple player boards and unrelated preview-like symbols.

### 5.3 Capture postprocessing

After scene composition, apply source-resolution changes, noninteger scaling, blur, sharpening, gamma/exposure, compression, video-like damage, foreground crops and chat/browser resampling. Apply geometric transforms consistently to lattice, masks, icon relations and visibility metadata.

The order matters. Blur before compositing differs from blur after compositing. A glow generated before resize differs from one applied after resize. Train a mixture informed by observed failure families.

Reflecting pieces exchanges J/L and S/Z; queue spatial order and coordinates must transform consistently. Do not use generic flip/rotation augmentation with unchanged labels.

### 5.4 Counterfactual render groups

For each latent state, render several independent backgrounds, palettes, skins, transparencies, grid settings and shadow styles. The semantic outputs should agree wherever both images carry the required evidence.

Then construct contrastive hard negatives: keep the same renderer/background but change one cell, one queue item, active/stack assignment or hold state. The model must notice the semantic change.

This creates two complementary training signals:

- Appearance can change while state remains the same.
- State can change while appearance settings remain the same.

A semantic consistency loss is more direct than hoping generic color jitter will eliminate shortcuts. Keep it on semantic outputs; forcing all hidden features to ignore appearance can remove information useful for visibility and ambiguity.

For transformed/cropped pairs, compare only corresponding observable targets and transform coordinates correctly. Pair consistency must not force confident agreement about content one image hides.

### 5.5 Honest recoverability and label integrity

Generate extreme transparency and occlusions to teach ambiguity, but do not treat fully invisible latent labels as confident reconstruction targets. Include paired states that produce indistinguishable observations, with an explicit ambiguity/review target.

Ground-truth state, visible masks and identifiability are distinct. A visible fragment can be insufficient to determine a piece; context can sometimes make a partly hidden fragment identifiable. Define this policy before training and use human review for real cases.

Do not use an unconstrained image generator to repaint labeled boards. It may add or remove cells or distort icons. It can supply backgrounds/assets, while a deterministic renderer retains control of state-bearing geometry.

The existing region generator is a useful historical reference, but is not a state-label oracle: it can omit rendered minos after constructing a pattern. New cell labels must correspond to the actual render decisions, with separate visual-fragment and logical-removal semantics.

Use an independent simple-skin oracle, render-label fixtures and family montages. Verify borders, row orientation, spawn clipping, transparency, active/ghost overlap, clear animations and repeated queue items explicitly.

## 6. Learning objectives and training from scratch

A concrete objective family is:

L = lambda_grid L_grid
  + lambda_locked L_locked
  + lambda_active L_active
  + lambda_ghost L_ghost
  + lambda_icons L_icons
  + lambda_order L_order
  + lambda_visibility L_visibility
  + lambda_pair L_pair

Use normalized geometric loss and boundary/anchor supervision for L_grid. L_locked is per-cell occupancy with density-balanced images. L_active includes presence/type/footprint. L_icons includes role, piece and local shape. L_order includes positions/relations and visible end. L_visibility includes component absence, occlusion and recoverability. L_pair enforces matched semantic consistency.

All weights are hyperparameters to tune on development data. No value is established by this research. Add deep supervision to bounded refinement steps, and introduce complex auxiliary losses only when controlled ablations show a benefit.

All-cell average loss can be dominated by easy empty cells. Balance board densities and inspect per-image worst-cell losses, queue length errors and active leakage. Hard-cell/hard-example weighting is worth testing; direct exact-state evaluation remains necessary.

### Fair training matrix

| Arm | Architecture and initialization | Purpose |
|---|---|---|
| A | Custom convolutional/attention lattice-query model, random weights | Main from-scratch candidate |
| A2 | Identical model with task-domain self-supervised pretraining | Tests a curriculum/pretraining benefit without external natural-image weights |
| B0 | Compatible transformer encoder + same custom decoder, random weights | Initialization control |
| B1 | Same B0 topology initialized from DINOv3 | Isolates external pretraining effect |
| C | Direct image-to-state sequence model, scratch | Tests whether explicit lattice structure is necessary |
| D | Current pipeline, unchanged | Reference floor and failure comparison; no runtime dependency |

Keep synthetic states, renderer families, development splits and allowed real data controlled. Report both equal-compute comparisons and adequately converged quality comparisons. Scratch training may need a longer schedule; a model stopped before convergence cannot decide the architecture question. Record total compute including pretraining.

A reasonable initial optimizer experiment is AdamW with warm-up, appropriate normalization for small batches, and separate learning-rate policies for any pretrained encoder. Tune from scratch rather than assuming fine-tuning hyperparameters transfer unchanged.

Start with roughly 10,000 examples to debug labels and learning. The first learning-curve study can compare 100,000 and one million distinct state/layout combinations with appearance variants generated on demand. These are budget hypotheses, not sufficiency claims. Track independent renderer/skin/state diversity and add difficult families based on development errors. TADA and domain-randomization research support targeted diversity rather than unbounded uniform expansion. [R10, R14-R16]

Maintain a mixture of representative common scenes and deliberately difficult counterexamples. Do not oversample severe effects so much that the model loses ordinary clean accuracy. Explore composition holdouts: train on transparency and no-grid separately, test their combination with unfamiliar backgrounds.

Full training should not begin until a bounded pilot measures memory and throughput. No GPU availability, training duration, cost or achieved accuracy is asserted here.

On this host, obey AGENTS: no unlimited memory; explicit commit/RAM ceiling within the default clamp; wall and step/sample budgets; bounded workers/cache/prefetch; durable logs/state; verified process tree. Use a small ceiling to characterize unknown peaks first. Normally keep a persistent job at or below 6 GiB when other work exists. Accelerator memory needs separate accounting. Do not disable engine guards or run a heavy release build alongside a large job.

## 7. Evaluation that tests the actual ambition

The main score is complete-state exact match, not region AP, cell mean accuracy or a favorable average of board/queue/hold scores.

A readable image succeeds only if the intended board, every required locked cell, active/stack separation, exact visible queue and hold are all correct. Evaluate active footprint separately if the consumer needs exact pose. Optional original locked-piece colors are separate: line clears, garbage and monochrome skins can make those identities unrecoverable.

At 99.9% independent cell accuracy, 200-cell exactness would be about 81.9%. A 95% exact-board probability in that simplified calculation needs approximately 99.9744% cell accuracy before queue/hold errors. Actual errors are correlated, so measure the joint event directly.

Predeclare a broad readable-screenshot population. Count abstentions as unsuccessful automatic parses in the 95% no-cleanup objective. Separately report ambiguous, unsupported and no-board inputs, with correct partial output/refusal and confidently wrong answers. Hidden content must not be silently counted as recovered.

### Required experiments

| Question | Measurement |
|---|---|
| Is geometry precise? | Cell-center/boundary error in fractions of a cell, exact cell assignment, and registration failure distribution |
| Does it need a grid? | Matched no-grid/grid render pairs and independent real no-grid skins |
| Is it color-dependent? | Palette permutation, monochrome, counterfactual recoloring and shape-identifiable hue conflicts |
| Does transparency generalize? | Independent background/alpha sweeps and real transparent-client captures |
| Can it handle missing components? | Empty versus absent versus cropped versus unreadable hold/queue, scored separately |
| Does active/ghost separation work? | Spawn clipping, resting/contact cases, tall neighboring towers, bright ghosts and clear transitions |
| Can queues vary? | Zero/long/repeated/wrapped queues, unknown middle slots, length extrapolation and visible-end mistakes |
| Does the custom design earn complexity? | Arms A/B/C exactness at converged quality and equal compute, plus ablations of lattice/context/refinement |
| Does deployment preserve it? | Same images through training runtime and exact released artifact, with whole-state and calibration comparison |

Begin with 300-500 independently reviewed real screenshots to establish baseline/error taxonomy. For a release claim, aim for roughly 2,000 or more held-out real images with enough independent sessions and renderer/skin families. Region-only labels do not replace full-state annotation.

Split related sessions, videos, replays, state ancestry and skin/renderer families together. Include unseen-client, unseen-layout and unseen-skin tests. Freeze a final test set; tune generation on development errors. Human ambiguity must be adjudicated rather than removed after seeing model results.

Report confidence intervals and session/family-aware resampling. A point estimate of 95% does not establish a 95% lower confidence bound. Statistical evidence on sampled domains does not guarantee arbitrary future domains.

Proposed quality gate: at least 95% full-state exact match over the readable population, including refusals as failures, with a strong release claim requiring the lower confidence bound to meet that target. Report every predeclared slice. Add a higher-confidence acceptance mode and its risk/coverage curve, without using rejection to replace the no-cleanup objective.

Confidence must be evaluated at state level. Softmax values or a product of cell confidences are not sufficient. Use an independent calibration set and inspect accepted-but-wrong examples. Calibrate after export/compression too. [R17-R18]

Accuracy is the first design criterion. Train and compare the accuracy-focused model before shrinking it for packaging. The existing local single-shot CPU goal remains a deployment gate: target warm p95 below 0.5 s and the documented 1 s budget on named hardware, with cold load and worst-case resources reported separately. If that is unmet, report the tradeoff and investigate compression; do not relabel a slower or less accurate model as satisfying both.

## 8. State contract and standalone tool

The public operation is conceptually:

parse_image(image) -> VisionState

A runtime package can expose Python, CLI and HTTP interfaces around the same checkpoint and decoder. It should require no skin selection, palette calibration, manually drawn boxes or another vision service for its normal path.

The rich output contains bottom-first locked rows, continuous grid geometry, active/ghost evidence, ordered queue slots, hold, visibility, uncertainty, model/schema versions and chosen player. Unknown is distinct from empty. An unreadable queue slot remains at its original index.

A simple interface fragment illustrates the semantics:

~~~json
{
  "queue": {
    "items": [
      {"piece": "T", "status": "observed"},
      {"piece": null, "status": "unknown"},
      {"piece": "I", "status": "observed"}
    ],
    "visibility": "cropped",
    "end_observed": false
  },
  "hold": {"status": "absent", "piece": null},
  "decision": "needs_review"
}
~~~

Queue means future previews; active is separate. The solver adapter must consume the current piece once, preserve bottom-first coordinates and refuse unsupported conversion of unknown cells/items into known emptiness.

Do not promise to reconstruct latent future pieces, invisible stack, lock timing, hold availability, combo or back-to-back unless observed or explicitly supplied. The user-facing result can be complete for all visible/identifiable components even when a cropped image cannot define a complete solver request.

Use fumen/blueprint adapters after state recognition. Occupied blocks may be exported as gray when original piece identity is unknown. Keep JSON as the authoritative rich representation.

During validation, show an overlay for grid assignment, locked/active/ghost evidence and queue ordering. A correction interface is useful product support; the acceptance target still measures images requiring no correction.

A Discord bot should be a thin caller of this parser. Parsing and expensive search have separate resource budgets. The same image must yield the same state across desktop/API/bot callers with the same model and settings.

## 9. Concrete implementation path

| Slice | Deliverable | Required gate |
|---|---|---|
| S0 | State/visibility annotation contract, full-state pilot, baseline receipt | Reviewed ground truth; exact-state failure taxonomy |
| S1 | Layered deterministic renderer and counterfactual render groups | Independent render-label oracle; family montages; transformed-coordinate integrity |
| S2 | Scratch custom encoder, lattice queries, cell/active heads | Learns clean synthetic cases from full images; precise lattice and no-grid tests |
| S3 | Queue/hold/role heads and joint refinement | Missing/cropped/long/repeated queues and complete-state synthetic gates |
| S4 | A/A2/B0/B1/C controlled study on real development sets | Converged whole-state comparison and unseen-family ablations |
| S5 | Released standalone model and solver/bot adapters | Sealed real-image gate; exact artifact, resource, CPU and user-path receipts |

Suggested code boundary is an independent model package with render/data/model/train/eval/export/runtime modules, plus a thin tetrguide adapter. A new tool may live beside existing vision code during development; it need not inherit its detector or palette logic. Existing serializers and basic libraries can be reused without inheriting recognition assumptions.

Source-only architecture implementation can follow this specification, but training must first pass host resource admission. No training is launched as part of this research follow-up.

Prior issues that implementation must preserve against:

- B-260: direct locked prediction, explicit active/ghost ambiguity, glow counterexamples and visible parse feedback.
- B-227: a single bottom-first coordinate adapter, pinned using asymmetric fields through the real product boundary.
- B-259: displayed state distinguishes observed, assumed, unknown and corrected fields.
- Existing opt-in flywheel: user images/corrections do not become unchecked training labels.
- B-102: future implementation commits name only task-owned paths.

New defects require BUG_LOG entries and named green gates. None is fixed by this research document.

## 10. Recommendation after the follow-up

The primary research candidate should be the scratch-trained custom lattice-query model. Pretrained initialization and a direct sequence model are serious experimental arms. The old detector/parser is only a baseline and does not define the new architecture's ceiling.

The most promising task-specific contributions are precise jointly trained lattice coordinates, cell queries that inspect shape rather than fixed center colors, separate semantic layers, explicit variable-length preview relationships, and counterfactual synthetic training that holds state fixed while appearance changes.

The strongest design will be selected by real complete-state results. Research supports these mechanisms; it does not establish that this particular combination already meets 95%. The remaining information limit is pixels that do not distinguish the requested states. Within identifiable inputs, the objective remains automatic complete parsing across broad visual variation.

## Evidence and references

This revision is a literature-and-source design review. It did not train models, reproduce paper results, benchmark competing tools on a common corpus or measure a new parser's accuracy. Recent preprints are mechanisms to test, not independent proof of product reliability.

Primary public sources:

- R1. Hou and Wang: [TABLET: Table Structure Recognition using Encoder-only Transformers](https://arxiv.org/abs/2506.07015), [method](https://arxiv.org/html/2506.07015v1). ICDAR 2025.
- R2. Hamdi et al.: [FastTab](https://arxiv.org/abs/2605.22422), [method](https://arxiv.org/html/2605.22422v1). May 2026 preprint; single-table/cropped formulation and grid-centric architecture.
- R3. Thomas et al.: [ConRTF](https://arxiv.org/abs/2607.00734), [method](https://arxiv.org/html/2607.00734v1). July 2026; arXiv states accepted ICDAR 2026.
- R4. Hamdi et al.: [TableSeq](https://arxiv.org/abs/2604.16070), [method](https://arxiv.org/html/2604.16070v1). April 2026.
- R5. Jaderberg et al.: [Spatial Transformer Networks](https://arxiv.org/abs/1506.02025). 2015/2016.
- R6. Zhu et al.: [Deformable DETR](https://arxiv.org/abs/2010.04159). ICLR 2021.
- R7. Peng et al.: [D-FINE](https://arxiv.org/abs/2410.13842). 2024.
- R8. Huang et al.: [Real-Time Object Detection Meets DINOv3 / DEIMv2](https://arxiv.org/abs/2509.20787). Reviewed January 2026 revision.
- R9. He, Girshick and Dollar: [Rethinking ImageNet Pre-training](https://arxiv.org/abs/1811.08883). 2018 report / ICCV 2019 work.
- R10. Nguyen et al.: [Do We Need All the Synthetic Data? Targeted Image Augmentation via Diffusion Models](https://arxiv.org/abs/2505.21574). Reviewed March 2026 revision.
- R11. Simeoni et al.: [DINOv3](https://arxiv.org/abs/2508.10104), [official repository](https://github.com/facebookresearch/dinov3). 2025 and current released variants.
- R12. Meta: [SAM 3 official repository](https://github.com/facebookresearch/sam3), including its March 2026 SAM 3.1 update.
- R13. Chen et al.: [Pix2Seq](https://arxiv.org/abs/2109.10852). ICLR 2022.
- R14. Tobin et al.: [Domain Randomization](https://arxiv.org/abs/1703.06907). 2017.
- R15. Hu et al.: [Learn2Synth](https://openaccess.thecvf.com/content/ICCV2025/html/Hu_Learn2Synth_Learning_Optimal_Data_Synthesis_Using_Hypergradients_for_Brain_Image_ICCV_2025_paper.html). ICCV 2025. Abstract/PDF exposed by search; direct HTML returned 403 in the initial review. Different domain.
- R16. Huang et al.: [Synthetic Object Compositions](https://arxiv.org/abs/2510.09110). January 2026 revision / CVPR 2026 listing.
- R17. Guo et al.: [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html). ICML 2017.
- R18. Geifman and El-Yaniv: [SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html). ICML 2019.
- R19. Microsoft: [ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html). Export/compression must be re-evaluated.
- R20. anthonys01: [snip2fumen](https://github.com/anthonys01/snip2fumen). Stated color/grid assumptions.
- R21. Trustacean and Yohpx: [Tetromino Detection](https://universe.roboflow.com/trustacean-and-yohpx/tetromino-detection). Owner's image count and detection annotations.
- R22. Max Moss: [project description](https://mxmoss.me/), [Sketris](https://mxmoss.me/sketris). Indexed author description; no broad exact-state benchmark established by this review.

Inspected repository evidence: AGENTS.md; docs/ROADMAP.md; docs/CONTINUATION_STATE.md; docs/EVIDENCE_POLICY.md; docs/BUG_LOG.md B-260/B-227/B-259/B-102; docs/DETECTOR_PLAN.md; backend/app/vision/ and backend/app/models.py; scripts/make_region_screenshots.py and scripts/benchmark_regions.py. Historical receipts retain their original artifact scope. No solver/detector historical gate is claimed as a fresh measurement here.
