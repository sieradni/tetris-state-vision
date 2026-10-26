# Local unlabeled screenshot stress set

The user authorized using the 166 images in
`C:/Users/micha/Code/tetrguide/data/raw_screenshots`, while explicitly warning
that correct board answers are uncertain. They are **unlabeled drafts**.
There are no annotations and no eligibility for training or accuracy scoring.
The original files are untouched; the standalone repository holds references.

Verified import/audit/profile evidence: `docs/evidence/real-drafts/scope.json`.
The collection contains 35,553,531 encoded bytes, no exact pixel duplicates and
seven aspect-preserving contact pages. These are file/raster facts, not proof of
independent sessions or client-family coverage. Per-frame names and thumbnails
remain in ignored local `data/` and `runs/` directories.

Local artifacts:

- `data/real-pilot-drafts/manifest.jsonl`: hashes, unresolved source groups,
  draft review status and false training/evaluation permission declarations.
- `runs/real-drafts-overview/overview-01.png` through `overview-07.png`:
  complete-frame contact sheets marked unlabeled.
- `runs/real-drafts-overview/profile.json`: dimensions, per-frame identities
  and page hashes, without inferred board state.

Representative contact pages 1, 4 and 7 were visually checked for complete
frames, readable draft IDs and preserved proportions. They show variable
scales/origins, decorative borders/backgrounds, text/animation over the playfield,
gray garbage styles, long previews, a piece above the board, and fragments of
neighboring panels/boards. These are qualitative observations of those pages,
not reviewed family labels or correctness judgments for all 166 images.

Useful work before board truth exists: inspect model failure modes qualitatively,
choose additional renderer families, identify layouts needing multi-board
association and select clear examples for future annotation. A model response
on these images remains a proposal. It cannot be counted as correct because it
looks plausible or agrees with the legacy parser. The synthetic pilot remains
the only input to the current optimizer.

Future labels may retain unknown cells/components where the raster is ambiguous.
A readable accuracy benchmark needs independent review of recoverable complete
states and declared session/ancestry/family splits; ambiguous examples have a
separate population. See [CORPUS.md](CORPUS.md). Do not force uncertain screenshots
into full board answers merely to fill a training set.
