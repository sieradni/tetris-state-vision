"""Adversarial integrity fixtures; none are real-image accuracy evidence."""
import copy
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

from PIL import Image, PngImagePlugin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import corpus_tools as corpus
from tetris_vision.state import Board, Cell, FallingPiece, Icon, Lattice, Queue, VisionState


class CorpusContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.images = self.root/"images"
        self.images.mkdir()
        for index in range(3):
            Image.new("RGB", (80, 160), (index*70, 20, 30)).save(self.images/f"{index}.png")
        self.output = self.root/"corpus"
        self.manifest = Path(corpus.init_corpus(self.images, self.output)["manifest"])
        self.header, self.rows = corpus.load_manifest(self.manifest, corpus.Limits())

    def write(self):
        self.manifest.write_text("".join(json.dumps(value)+"\n" for value in (self.header, *self.rows)), encoding="utf-8")

    def review(self, index=0, split="test"):
        row = self.rows[index]
        state = VisionState(Board(tuple(tuple(Cell(False) for _ in range(10)) for _ in range(20)),
                                  Lattice((10, 150), (6, 0), (0, -6))),
                            FallingPiece("absent"), FallingPiece("absent"),
                            Queue((), "absent", False, False), Icon("absent"), "accepted",
                            producer="human-annotation")
        path = self.output/"annotations"/f"{index}.json"
        path.write_text(state.to_json(), encoding="utf-8")
        row.update(annotation=path.name, annotation_sha256=corpus.digest_file(path),
                   population="readable", split=split,
                   groups={key: f"{key}-{index}" for key in (*corpus.GROUPS, "client_family")},
                   permission={"training": True, "evaluation": True, "evidence": "fixture permission declaration"},
                   review={"status": "reviewed", "annotator": "fixture-annotator", "annotator_kind": "human",
                           "reviewer": "fixture-reviewer", "reviewer_kind": "human",
                           "reviewed_at": "2026-10-26T10:00:00-07:00", "evidence": "fixture independent review declaration"})
        return row, path

    def audit(self):
        self.write()
        return corpus.audit(self.manifest)

    def assert_invalid(self, phrase):
        result = self.audit()
        self.assertTrue(any(phrase in error["reason"] for error in result.errors), result.report())
        self.assertEqual(result.report()["status"], "failed")

    def amend_annotation(self, path, row, mutate):
        document = json.loads(path.read_text())
        mutate(document)
        path.write_text(json.dumps(document), encoding="utf-8")
        row["annotation_sha256"] = corpus.digest_file(path)

    def test_import_is_draft_only_and_preserves_source_and_existing_output(self):
        before = [corpus.digest_file(path) for path in sorted(self.images.iterdir())]
        result = self.audit()
        self.assertFalse(result.errors)
        self.assertEqual(result.report()["drafts"], 3)
        self.assertFalse(result.eligible)
        self.assertTrue(all(not row["permission"]["training"] and row["annotation"] is None for row in self.rows))
        with self.assertRaises(FileExistsError):
            corpus.init_corpus(self.images, self.output)
        self.assertEqual(before, [corpus.digest_file(path) for path in sorted(self.images.iterdir())])

    def test_reviewed_permitted_record_is_admitted(self):
        row, _ = self.review()
        result = self.audit()
        self.assertFalse(result.errors, result.report())
        self.assertEqual(result.eligible, {row["sample_id"]: "test"})

    def test_permission_is_specific_to_use(self):
        row, _ = self.review(split="train")
        row["permission"]["training"] = False
        result = self.audit()
        self.assertFalse(result.errors)
        self.assertFalse(result.eligible)
        self.assertIn("training permission absent", result.warnings[0]["reason"])
        row["split"] = "test"
        self.assertEqual(len(self.audit().eligible), 1)

    def test_permission_claim_requires_evidence(self):
        row, _ = self.review()
        row["permission"]["evidence"] = None
        self.assert_invalid("permission evidence required")

    def test_model_proposal_cannot_self_promote(self):
        row, path = self.review()
        row["review"]["annotator_kind"] = "model"
        self.amend_annotation(path, row, lambda d: d.update(producer="model", model_id="weights-sha"))
        self.assert_invalid("separate from model/synthetic proposals")
        row["review"]["status"] = "draft"
        result = self.audit()
        self.assertFalse(result.errors)
        self.assertFalse(result.eligible)

    def test_review_is_independent_human_and_timestamped(self):
        row, _ = self.review()
        for field, value, reason in (("reviewer_kind", "agent", "independent human"),
                                     ("reviewer", "fixture-annotator", "must differ"),
                                     ("reviewed_at", "2026-10-26T10:00:00", "timezone"),
                                     ("evidence", "", "review evidence")):
            with self.subTest(field=field):
                previous = row["review"][field]
                row["review"][field] = value
                self.assert_invalid(reason)
                row["review"][field] = previous

    def test_unresolved_groups_are_never_eligible(self):
        row, _ = self.review()
        row["groups"]["session"] = "unknown"
        self.assert_invalid("resolve grouping")

    def test_group_leakage_for_each_required_axis(self):
        first, _ = self.review(0, "train")
        second, _ = self.review(1, "test")
        for key in corpus.GROUPS:
            with self.subTest(axis=key):
                previous = second["groups"][key]
                second["groups"][key] = first["groups"][key]
                self.assert_invalid(f"{key} leakage")
                second["groups"][key] = previous

    def test_same_group_within_one_split_and_client_holdout(self):
        first, _ = self.review(0, "train")
        second, _ = self.review(1, "train")
        second["groups"]["session"] = first["groups"]["session"]
        self.assertFalse(self.audit().errors)
        self.header["client_holdouts"][first["groups"]["client_family"]] = "test"
        self.assert_invalid("client holdout")

    def test_duplicate_pixels_across_encodings_cannot_inflate_sample_count(self):
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("encoding-note", "same raster, different file")
        with Image.open(self.images/"0.png") as image:
            image.save(self.images/"1.png", pnginfo=metadata)
        for index in (0, 1):
            row, _ = self.review(index)
            row["image_sha256"], row["pixel_sha256"], _ = corpus.image_identity(self.images/f"{index}.png", corpus.Limits())
        self.assertNotEqual(self.rows[0]["image_sha256"], self.rows[1]["image_sha256"])
        self.assertEqual(self.rows[0]["pixel_sha256"], self.rows[1]["pixel_sha256"])
        self.assert_invalid("duplicate eligible pixels")

    def test_invisible_rgb_is_not_an_independent_image(self):
        for index, color in enumerate(((0, 0, 0, 0), (255, 90, 20, 0))):
            Image.new("RGBA", (20, 20), color).save(self.images/f"{index}.png")
        first = corpus.image_identity(self.images/"0.png", corpus.Limits())
        second = corpus.image_identity(self.images/"1.png", corpus.Limits())
        self.assertNotEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])

    def test_modified_image_and_annotation_are_detected(self):
        row, path = self.review()
        original = path.read_bytes()
        path.write_bytes(original+b" ")
        self.assert_invalid("annotation SHA256 mismatch")
        path.write_bytes(original)
        Image.new("RGB", (80, 160), "red").save(self.images/"0.png")
        self.assert_invalid("image byte/pixel SHA256 mismatch")

    def test_geometry_registration_and_image_bounds(self):
        row, path = self.review()
        original = json.loads(path.read_text())
        self.amend_annotation(path, row, lambda d: d["board"].update(lattice=None))
        self.assert_invalid("geometry")
        path.write_text(json.dumps(original))
        self.amend_annotation(path, row, lambda d: d["board"]["lattice"].update(origin=[-10, 150]))
        self.assert_invalid("outside image")
        path.write_text(json.dumps(original))
        self.amend_annotation(path, row, lambda d: d.update(active={"status": "observed", "piece": "I", "cells": [[30, 2], [31, 2], [32, 2], [33, 2]]}))
        self.assert_invalid("piece lies outside")

    def test_ambiguous_partial_board_cannot_label_off_image_cells(self):
        row, path = self.review()
        row["population"] = "ambiguous"
        self.amend_annotation(path, row, lambda d: d.update(decision="needs_review"))
        self.amend_annotation(path, row, lambda d: d["board"]["lattice"].update(origin=[-2, 150]))
        self.assert_invalid("known canonical cell lies outside")
        # The same visible raster cannot establish either empty or occupied
        # off-image cells. Both certain claims are rejected by the same check.
        self.amend_annotation(path, row, lambda d: d["board"]["rows"][0][0].update(occupied=True))
        self.assert_invalid("known canonical cell lies outside")
        def mask_unseen(document):
            for cells in document["board"]["rows"]:
                cells[0] = {"occupied": None, "semantic": None, "visibility": "cropped"}
        self.amend_annotation(path, row, mask_unseen)
        self.assertFalse(self.audit().errors)

    def test_legacy_schema_and_wrong_population_are_rejected(self):
        row, path = self.review()
        self.amend_annotation(path, row, lambda d: d.update(schema_version="1.0"))
        self.assert_invalid("no silent legacy migration")
        self.amend_annotation(path, row, lambda d: d.update(schema_version="1.1"))
        row["population"] = "ambiguous"
        self.assert_invalid("population/annotation")

    def test_path_escape_is_refused(self):
        self.rows[0]["image"] = "../corpus/manifest.jsonl"
        self.assert_invalid("escapes declared root")

    def test_record_asset_pixel_and_line_budgets(self):
        for limits, reason in ((corpus.Limits(records=2), "record budget"),
                                (corpus.Limits(asset_bytes=1), "asset-byte budget")):
            with self.subTest(reason=reason), self.assertRaisesRegex(corpus.CorpusError, reason):
                corpus.init_corpus(self.images, self.root/reason, limits)
            self.assertFalse((self.root/reason).exists())
        with self.assertRaisesRegex(corpus.CorpusError, "pixel budget"):
            corpus.image_identity(self.images/"0.png", corpus.Limits(image_pixels=10))
        with self.assertRaisesRegex(corpus.CorpusError, "record budget"):
            corpus.audit(self.manifest, corpus.Limits(records=2))
        self.manifest.write_text(json.dumps(self.header)+"\n"+" "*65537)
        with self.assertRaisesRegex(corpus.CorpusError, "line budget"):
            corpus.audit(self.manifest)

    def test_malformed_rows_report_failure_without_crashing(self):
        for malformed in (None, [], {}, {"review": []}, {**self.rows[0], "review": None}):
            with self.subTest(row=malformed):
                self.rows = [malformed]
                self.assert_invalid("record")

    def test_duplicate_ids_do_not_inflate_admitted_count(self):
        self.review()
        self.rows.append(copy.deepcopy(self.rows[0]))
        result = self.audit()
        self.assertEqual(result.report()["eligible_by_split"]["test"], 1)
        self.assertTrue(any("duplicate sample id" in error["reason"] for error in result.errors))

    def test_strict_json_rejects_duplicate_fields_nonfinite_and_nonobjects(self):
        for raw in ('{"split":"train","split":"test"}', '{"value":NaN}'):
            with self.assertRaises(corpus.CorpusError):
                corpus.parse_json(raw)
        self.manifest.write_text("null\n")
        with self.assertRaisesRegex(corpus.CorpusError, "header must be an object"):
            corpus.audit(self.manifest)

    def test_seal_requires_review_permission_and_external_identity(self):
        snapshot = self.root/"seal.json"
        self.rows[0]["split"] = "test"
        self.write()
        with self.assertRaisesRegex(corpus.CorpusError, "reviewed and permitted"):
            corpus.freeze(self.manifest, snapshot)
        self.assertFalse(snapshot.exists())
        self.review()
        self.write()
        receipt = corpus.freeze(self.manifest, snapshot)
        with self.assertRaises(FileExistsError):
            corpus.freeze(self.manifest, snapshot)
        with self.assertRaisesRegex(corpus.CorpusError, "external expected"):
            corpus.verify_snapshot(self.manifest, snapshot, None)
        self.assertEqual(corpus.verify_snapshot(self.manifest, snapshot, receipt["snapshot_sha256"])["status"], "verified")
        snapshot.write_text(snapshot.read_text()+" ")
        with self.assertRaisesRegex(corpus.CorpusError, "byte identity changed"):
            corpus.verify_snapshot(self.manifest, snapshot, receipt["snapshot_sha256"])

    def test_seal_rejects_record_changes_but_allows_unrelated_training_additions(self):
        row, _ = self.review()
        self.write()
        snapshot = self.root/"seal.json"
        sha = corpus.freeze(self.manifest, snapshot)["snapshot_sha256"]
        self.review(1, "train")
        self.write()
        self.assertEqual(corpus.verify_snapshot(self.manifest, snapshot, sha)["status"], "verified")
        row["source"]["reference"] = "changed declaration"
        self.write()
        with self.assertRaisesRegex(corpus.CorpusError, "frozen records changed"):
            corpus.verify_snapshot(self.manifest, snapshot, sha)

    def test_new_training_leak_invalidates_existing_seal(self):
        row, _ = self.review()
        self.write()
        snapshot = self.root/"seal.json"
        sha = corpus.freeze(self.manifest, snapshot)["snapshot_sha256"]
        training, _ = self.review(1, "train")
        training["groups"]["ancestry"] = row["groups"]["ancestry"]
        self.write()
        with self.assertRaisesRegex(corpus.CorpusError, "current corpus audit failed"):
            corpus.verify_snapshot(self.manifest, snapshot, sha)

    def run_cli(self, *arguments, environment=None):
        return subprocess.run([sys.executable, str(Path(corpus.__file__)), *map(str, arguments)],
                              capture_output=True, text=True, timeout=15, env=environment)

    def test_actual_cli_under_named_job_audits_seals_and_verifies(self):
        self.review()
        self.write()
        check = self.run_cli("audit", self.manifest)
        self.assertEqual(check.returncode, 0, check.stderr)
        self.assertEqual(json.loads(check.stdout)["eligible_by_split"]["test"], 1)
        snapshot = self.root/"seal.json"
        seal = self.run_cli("freeze", self.manifest, "--snapshot", snapshot)
        self.assertEqual(seal.returncode, 0, seal.stderr)
        sha = json.loads(seal.stdout)["snapshot_sha256"]
        verified = self.run_cli("verify", self.manifest, "--snapshot", snapshot, "--expected-sha256", sha)
        self.assertEqual(verified.returncode, 0, verified.stderr)
        self.assertEqual(json.loads(verified.stdout)["status"], "verified")

    def test_cli_missing_job_identity_refuses_before_corpus_output(self):
        environment = dict(os.environ)
        environment.pop("TETRIS_VISION_JOB_NAME", None)
        output = self.root/"not-created"
        result = self.run_cli("init", "--image-root", self.images, "--output", output, environment=environment)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(output.exists())
        self.assertIn("Job", result.stderr)

    def test_nonobject_annotation_and_malformed_declarations_fail_closed(self):
        row, path = self.review()
        path.write_text("null", encoding="utf-8")
        row["annotation_sha256"] = corpus.digest_file(path)
        self.assert_invalid("annotation must be an object")
        malformed = self.run_cli("audit", self.manifest)
        self.assertEqual(malformed.returncode, 1)
        self.assertEqual(json.loads(malformed.stdout)["status"], "failed")


if __name__ == "__main__":
    unittest.main()
