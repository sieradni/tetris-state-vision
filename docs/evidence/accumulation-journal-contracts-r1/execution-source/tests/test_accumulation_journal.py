"""Pure cursor and optimizer-ownership checks; no neural imports."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from accumulation_contract import group_specs
from accumulation_journal import execute_group, group_entry, require_optimizer_ownership, validate_ledger, verify_resume_cursor


def recipes(policy, update):
    curriculum = "all-class-fit-v1" if policy == "ordered-eight-v1" else "all-class-interleaved-v1"
    return [{"sample_id": f"all-class-{index:02d}-v{view}", "group_id": f"all-class-{index:02d}",
             "index": index, "view": view, "split": "fit", "curriculum": curriculum, "profile_square": False,
             "image_pixels_sha256": "a"*64, "state_sha256": "b"*64}
            for _, index, view in group_specs(policy, update)]


def entry(policy, update):
    return group_entry(policy, update, recipes(policy, update), mean_loss=1.25, gradient_norm=.5)


class FakeModel:
    def __init__(self, parameters):
        self._parameters = parameters

    def parameters(self):
        return iter(self._parameters)


class FakeOptimizer:
    def __init__(self, groups):
        self.param_groups = [{"params": group} for group in groups]


class AccumulationJournalTest(unittest.TestCase):
    def test_explicit_group_cursor_and_complete_budget(self):
        for policy, count, size in (("online-interleaved-v1", 1024, 1), ("balanced-eight-v1", 128, 8),
                                    ("ordered-eight-v1", 128, 8)):
            rows = [entry(policy, i) for i in range(1, count+1)]
            partial = validate_ledger(policy, rows[:3])
            self.assertEqual((partial["images_seen"], partial["optimizer_updates"], partial["next_image"],
                              partial["next_update"]), (3*size, 3, 3*size+1, 4))
            final = validate_ledger(policy, rows)
            self.assertEqual((final["images_seen"], final["optimizer_updates"], final["status"], final["next_image"]),
                             (1024, count, "complete", None))

    def test_partial_group_and_bad_order_refuse(self):
        policy = "balanced-eight-v1"
        with self.assertRaisesRegex(ValueError, "complete group"):
            group_entry(policy, 1, recipes(policy, 1)[:3], mean_loss=1., gradient_norm=1.)
        rows = [entry(policy, 1), entry(policy, 2)]
        rows[1]["images"][0]["image"] = 99
        with self.assertRaises(ValueError):
            validate_ledger(policy, rows)
        with self.assertRaises(ValueError):
            validate_ledger(policy, [entry(policy, 2)])

    def test_cursor_mismatch_and_partial_snapshot_refuse(self):
        policy = "balanced-eight-v1"
        rows = [entry(policy, 1), entry(policy, 2)]
        self.assertEqual(verify_resume_cursor(policy, rows, snapshot_images=16, snapshot_updates=2)["next_image"], 17)
        for images, updates in ((15, 2), (16, 1), (24, 3), (True, 2)):
            with self.subTest(images=images, updates=updates), self.assertRaises(ValueError):
                verify_resume_cursor(policy, rows, snapshot_images=images, snapshot_updates=updates)

    def test_recipe_identity_nan_and_bool_are_rejected(self):
        policy = "ordered-eight-v1"
        for mutation in (lambda r: r.update(index=True), lambda r: r.update(state_sha256="bad"),
                         lambda r: r.update(curriculum="all-class-interleaved-v1"),
                         lambda r: r.update(profile_square=True), lambda r: r.update(extra=float("nan"))):
            rows = recipes(policy, 1)
            mutation(rows[0])
            with self.assertRaises((ValueError, TypeError)):
                group_entry(policy, 1, rows, mean_loss=1., gradient_norm=1.)
        with self.assertRaises(ValueError):
            group_entry(policy, 1, recipes(policy, 1), mean_loss=float("nan"), gradient_norm=0.)

    def test_entry_freezes_recipe_and_survives_json_roundtrip(self):
        policy = "balanced-eight-v1"
        source = recipes(policy, 1)
        source[0]["nested"] = {"values": [1, 2]}
        saved = group_entry(policy, 1, source, mean_loss=1., gradient_norm=2.)
        source[0]["nested"]["values"].append(3)
        self.assertEqual(saved["images"][0]["recipe"]["nested"]["values"], [1, 2])
        self.assertEqual(validate_ledger(policy, json.loads(json.dumps([saved])))["images_seen"], 8)

    def test_real_curated_recipe_shape_is_accepted_without_filling_labels(self):
        path = ROOT / "docs/evidence/m2-stream-headrate-control-s3/training-probe.jsonl"
        by_key = {(r["index"], r["view"]): r for r in map(json.loads, path.read_text().splitlines())}
        spec = group_specs("ordered-eight-v1", 1)
        actual = [copy.deepcopy(by_key[index, view]) for _, index, view in spec]
        result = group_entry("ordered-eight-v1", 1, actual, mean_loss=0., gradient_norm=0.)
        self.assertEqual([row["recipe"] for row in result["images"]], actual)
        self.assertTrue(all("target_type" not in row["recipe"] for row in result["images"]))

    def test_optimizer_ownership_rejects_missing_extra_and_duplicates(self):
        a, b, c = object(), object(), object()
        model = FakeModel([a, b])
        require_optimizer_ownership(model, FakeOptimizer([[b], [a]]))
        for groups in ([[a]], [[a, b, c]], [[a, a]], [[], []]):
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                require_optimizer_ownership(model, FakeOptimizer(groups))
        with self.assertRaises(ValueError):
            require_optimizer_ownership(FakeModel([]), FakeOptimizer([]))

    def test_execution_preflights_and_only_returns_complete_update(self):
        a, b = object(), object()
        model, optimizer = FakeModel([a, b]), FakeOptimizer([[a, b]])
        policy = "balanced-eight-v1"
        source = recipes(policy, 1)
        with patch("accumulation_journal.apply_group", return_value={"mean_loss": 2., "gradient_norm": 3., "images": 8}) as apply:
            result = execute_group(model, optimizer, policy, 1, source, lambda _: None)
            self.assertEqual(validate_ledger(policy, [result])["images_seen"], 8)
            self.assertEqual(apply.call_count, 1)
            invalid = copy.deepcopy(source)
            invalid.pop()
            with self.assertRaises(ValueError):
                execute_group(model, optimizer, policy, 1, invalid, lambda _: None)
            with self.assertRaises(ValueError):
                execute_group(model, FakeOptimizer([[a]]), policy, 1, source, lambda _: None)
            self.assertEqual(apply.call_count, 1)
        with patch("accumulation_journal.apply_group", side_effect=RuntimeError("interrupted")):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                execute_group(model, optimizer, policy, 1, source, lambda _: None)
        with patch("accumulation_journal.apply_group", return_value={"mean_loss": 2., "gradient_norm": 3., "images": 3}):
            with self.assertRaisesRegex(ValueError, "complete optimizer"):
                execute_group(model, optimizer, policy, 1, source, lambda _: None)


if __name__ == "__main__":
    unittest.main()
