# Reviewed real-image corpus

`scripts/corpus_tools.py` prepares and checks local screenshot collections.
It does not annotate images, grant permission, or establish screenshot accuracy.
The test fixtures are invented integrity examples, never benchmark data.

Use the existing bounded supervisor for every invocation. The CLI verifies exact
membership in its named Windows Job before opening the corpus. For example:

```powershell
python scripts/run_bounded.py --memory-mib 512 --wall-seconds 180 --work-items 500 --receipt runs/corpus-init.json -- .venv/Scripts/python.exe scripts/corpus_tools.py --max-records 500 --max-asset-mib 1024 init --image-root C:/path/to/screenshots --output data/real-pilot
python scripts/run_bounded.py --memory-mib 512 --wall-seconds 180 --work-items 500 --receipt runs/corpus-audit.json -- .venv/Scripts/python.exe scripts/corpus_tools.py audit data/real-pilot/manifest.jsonl
```

These are argument examples; choose an actual source explicitly and use new
receipt/output paths. Long commands launch hidden and detached with logs and a
continuation checkpoint as required by AGENTS.md. The child admits at most 500
records, 1 GiB aggregate encoded assets, 64 MiB per image, 8 million decoded
pixels per image and 1 MiB per annotation. Manifest size is at most 16 MiB and
each JSONL line at most 64 KiB. Limits may be lowered. Directory initialization
scans at most 10,000 entries, accepts only flat PNG/JPG/JPEG files and refuses
oversized collections instead of silently selecting a subset. Original files
are referenced, never modified or copied; output directories are exclusive.
Animations must be extracted into frames with shared session/ancestry first.

The first JSONL line declares format/version, absolute image and annotation
roots, mandatory grouping axes, and explicit client-family holdouts. Remaining
lines are records. Initialization creates draft records, null annotations,
unresolved groups, an unclassified population and false training/evaluation
permission. Keep image roots private: this manifest is local data, ignored by Git.

Before a record becomes eligible:

1. Record the real source reference and rights/consent evidence separately for
   training and evaluation. Only permitted uses are eligible; neither is inferred
   from file availability or from a label correction.
2. Resolve capture session, replay/video/state ancestry, skin family, renderer
   family and client family. Related frames and re-encodings are not independent
   examples. All eligible members of each mandatory group stay in one split.
   Declare client-family holdouts in the header. Do not invent new group IDs to
   make a split pass. This tool validates declared IDs, not their truth.
3. Write a current schema 1.1 image-supported annotation following
   [ANNOTATION.md](ANNOTATION.md). Select `readable`, `ambiguous`, `no_board` or
   `unsupported`; its decision must respectively be `accepted`, `needs_review`,
   `no_board` or `unsupported`. Unknown occupancy is not empty. Geometry and
   observed canonical pieces must fit the visible source. Legacy annotations
   require explicit re-review; they are not migrated during admission.
4. An independent human reviewer records identity, timezone-bearing review time
   and review evidence. The annotator may be a human, agent or model; reviewer
   and annotator identities must differ. Review produces a separate ground-truth
   document with producer `human-annotation`, `human-reviewed` or
   `human-corrected` and null model ID. Preserve proposals separately. Merely
   changing a manifest flag cannot admit an uncorrected model document.
5. Record computed encoded-image, canonical-pixel and annotation SHA256 values.
   Canonical pixel identity covers dimensions, black-composited RGB and alpha;
   encoding metadata and invisible RGB cannot inflate independent image counts.
   Exact pixel duplicates among eligible records fail. Draft duplicates warn.

Review and permission evidence are declarations with an audit trail, not a
cryptographic proof of human review or labeling correctness. The auditor cannot
detect dishonest group IDs, perceptually similar but unequal frames, or incorrect
human judgments. Independent sample review and visual error analysis remain
required before scientific evaluation. Draft records never become eligible even
when their fields resemble a complete label.

Use `freeze MANIFEST --snapshot NEW.json --splits calibration test` to seal the
chosen held-out records after audit. Every selected record must be eligible and
each chosen split nonempty. Pin the returned **snapshot SHA256 outside the
snapshot**, in an experiment receipt. Use `verify MANIFEST --snapshot FILE
--expected-sha256 SHA` before evaluation. Verification checks current assets,
review, permission, declared split policy and the selected record identities.
Adding unrelated training records is allowed; introducing a shared held-out
group fails the global audit. Changing labels, images, review declarations,
held-out membership or the split policy requires a new explicitly versioned
benchmark. Moving the roots with byte-identical assets does not change the seal.
No existing snapshot is overwritten. This is immutable identity checking, not
operating-system write protection.

The current optimizer driver still consumes only its small deterministic
synthetic pilot. It does not consume this manifest or any real image. Real-data
training requires a separate acceptance-gated integration and a reviewed corpus.

The user-approved local 166-image folder is now imported as drafts only; its
purpose, private overview and verified aggregate receipt are in
[REAL_STRESS_SET.md](REAL_STRESS_SET.md). Board answers are explicitly uncertain.
