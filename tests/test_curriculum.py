from collections import Counter
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from m2_curriculum import (ALL_CLASS, INTERLEAVED, STREAM, distinct_training_groups, probe_specs,
                           replay_case, sample_for, training_case, training_spec, validate_curriculum)
from m2_stream_data import sample_case
from m2_interleaved_order import cycle_order, distinct_groups, interleaved_spec


class CurriculumTests(unittest.TestCase):
    def test_interleaved_training_preserves_exact_old_pixels_labels_and_probe_order(self):
        actual_order = []
        for step in range(1, 33):
            index, view = interleaved_spec(step)
            self.assertEqual(training_spec(step, INTERLEAVED), (index, view))
            actual = training_case(step, 20, curriculum=INTERLEAVED)
            original = sample_for(index, view, "training", 20, curriculum=ALL_CLASS)
            self.assertEqual(actual.frame.image.tobytes(), original.frame.image.tobytes())
            self.assertEqual(actual.frame.state.to_json(), original.frame.state.to_json())
            self.assertEqual({key: value for key, value in actual.record.items() if key != "curriculum"},
                             {key: value for key, value in original.record.items() if key != "curriculum"})
            self.assertEqual(actual.record["curriculum"], INTERLEAVED)
            actual_order.append((index, view))
        self.assertEqual(tuple(actual_order), cycle_order(0))
        for split in ("training", "development"):
            self.assertEqual(probe_specs(split, 3, INTERLEAVED), probe_specs(split, 3, ALL_CLASS))
        for cursor in (0, 1, 7, 8, 9, 31, 32, 33, 640, 1024, 4096):
            self.assertEqual(distinct_training_groups(cursor, INTERLEAVED), distinct_groups(cursor))

    def test_ordered_and_interleaved_recipes_require_their_declared_policy(self):
        for policy, other in ((ALL_CLASS, INTERLEAVED), (INTERLEAVED, ALL_CLASS)):
            case = sample_for(0, 0, "training", 20, curriculum=policy)
            record = json.loads(json.dumps(case.record))
            self.assertEqual(replay_case(record, split="training", curriculum=policy).record, case.record)
            with self.assertRaises(ValueError):
                replay_case(record, split="training", curriculum=other)
        self.assertEqual(training_spec(1, INTERLEAVED, profile_square=True), (0, 0))
        self.assertEqual(training_spec(2, INTERLEAVED, profile_square=True), (0, 0))
        with self.assertRaises(ValueError):
            validate_curriculum(INTERLEAVED, 64)

    def test_stream_default_keeps_original_records_pixels_and_global_order(self):
        for step in (1, 2, 3, 4, 17):
            group, view = training_spec(step)
            self.assertEqual((group, view), ((step - 1) // 3, (step - 1) % 3))
            actual = training_case(step, 20)
            expected = sample_case(group, view, "training", 20)
            self.assertEqual(actual.record, expected.record)
            self.assertEqual(actual.frame.image.tobytes(), expected.frame.image.tobytes())
        actual = sample_for(2, 2, "training", 20, component_capacity=64)
        self.assertEqual(actual.record, sample_case(2, 2, "training", 20, final_capture=True).record)

    def test_fit_cycle_is_balanced_and_reuses_the_same_sixteen_ancestries(self):
        counts = Counter()
        for step in range(1, 33):
            self.assertEqual(training_spec(step, ALL_CLASS), ((step - 1) // 2, (step - 1) % 2))
            case = training_case(step, 20, curriculum=ALL_CLASS)
            role = case.frame.state.active
            counts[role.piece if role.status == "observed" else role.status] += 1
        self.assertEqual(counts, {**{piece: 4 for piece in "IOTSZJL"}, "absent": 4})
        self.assertEqual(training_case(1, 20, curriculum=ALL_CLASS).record, training_case(33, 20, curriculum=ALL_CLASS).record)
        self.assertEqual([distinct_training_groups(step, ALL_CLASS) for step in (0, 1, 2, 3, 32, 64, 4096)], [0, 1, 1, 2, 16, 16, 16])

    def test_probe_population_and_independent_development_are_explicit(self):
        specs = probe_specs("training", 3, ALL_CLASS)
        self.assertEqual(len(specs), 32)
        self.assertEqual(len(set(specs)), 32)
        self.assertEqual(len(probe_specs("training", 3)), 9)
        self.assertEqual(len(probe_specs("development", 6, ALL_CLASS)), 18)
        train = sample_for(0, 0, "training", 20, curriculum=ALL_CLASS)
        dev = sample_for(0, 0, "development", 20, curriculum=ALL_CLASS)
        self.assertEqual(dev.record, sample_case(0, 0, "development", 20).record)
        self.assertNotEqual(train.record["group_id"], dev.record["group_id"])
        self.assertNotEqual(train.record["image_pixels_sha256"], dev.record["image_pixels_sha256"])

    def test_complete_record_replay_refuses_policy_or_raster_or_metadata_changes(self):
        for curriculum in (STREAM, ALL_CLASS, INTERLEAVED):
            for split in ("training", "development"):
                case = sample_for(0, 0, split, 20, curriculum=curriculum)
                record = json.loads(json.dumps(case.record))
                actual = replay_case(record, split=split, curriculum=curriculum)
                self.assertEqual(actual.frame.image.tobytes(), case.frame.image.tobytes())
                for changed in ({**record, "image_pixels_sha256": "0" * 64}, {**record, "unexpected": True}):
                    with self.assertRaises(ValueError):
                        replay_case(changed, split=split, curriculum=curriculum)
        fit = json.loads(json.dumps(training_case(1, 20, curriculum=ALL_CLASS).record))
        with self.assertRaises(ValueError):
            replay_case(fit, split="training", curriculum=STREAM)
        with self.assertRaises(ValueError):
            replay_case(fit, split="development", curriculum=ALL_CLASS)

    def test_two_square_steps_and_invalid_policy_capacity_and_cursors(self):
        for policy in (STREAM, ALL_CLASS, INTERLEAVED):
            cases = [training_case(step, 20, curriculum=policy, profile_square=True) for step in (1, 2)]
            self.assertEqual(cases[0].frame.image.size, (512, 512))
            self.assertEqual(cases[0].record, cases[1].record)
            self.assertTrue(cases[1].record["profile_square"])
            for step in (0, True, 4097):
                with self.assertRaises(ValueError):
                    training_spec(step, policy)
            with self.assertRaises(ValueError):
                training_spec(3, policy, profile_square=True)
        for policy, capacity in (("unknown", 0), (False, 0), (ALL_CLASS, 64), (STREAM, True), (STREAM, 9)):
            with self.assertRaises(ValueError):
                validate_curriculum(policy, capacity)
        for changed in ({"split": "test"}, {"seed": True}, {"profile_square": 1}):
            kwargs = {"group": 0, "view": 0, "split": "training", "seed": 20, **changed}
            with self.assertRaises(ValueError):
                sample_for(**kwargs)


if __name__ == "__main__":
    unittest.main()
