"""Contracts for the canonical development extension used by ordinary replays."""
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from probe_stream_errors import CANONICAL_DEVELOPMENT_GROUPS, extension_specs
from record_stream_errors import verify_extension

SEED = 20261026


def saved_recipes(groups=3):
    return [{"sample_id": f"development-{SEED+group+10_000_000}-v{view}",
             "group_id": f"ancestry-development-{SEED+group+10_000_000}", "split": "development",
             "group_index": group, "view": view, "base_seed": SEED, "state_sha256": "0"*64}
            for group in range(groups) for view in range(3)]


def metadata(recorded=3):
    specs = extension_specs(SEED, recorded, CANONICAL_DEVELOPMENT_GROUPS)
    recipes = [{"sample_id": spec["sample_id"], "group_id": spec["group_id"], "split": "development",
                "group_index": spec["group_index"], "view": spec["view"], "base_seed": spec["base_seed"],
                "state_sha256": "0"*64} for spec in specs]
    return {"saved_groups": recorded, "evaluated_groups": CANONICAL_DEVELOPMENT_GROUPS,
            "canonical_groups": CANONICAL_DEVELOPMENT_GROUPS, "base_seed": SEED,
            "saved_cases": recorded*3, "extension_cases": len(specs), "recipes": recipes}


class DevelopmentExtensionContractTests(unittest.TestCase):
    def test_canonical_set_is_six_groups_and_extends_as_a_prefix(self):
        self.assertEqual(CANONICAL_DEVELOPMENT_GROUPS, 6)
        specs = extension_specs(SEED, 3, 6)
        self.assertEqual(len(specs), 9)
        self.assertEqual([spec["group_index"] for spec in specs], [3]*3+[4]*3+[5]*3)
        self.assertEqual(specs[0]["sample_id"], "development-30261029-v0")
        self.assertEqual(specs[0]["group_id"], "ancestry-development-30261029")
        self.assertEqual(specs[0]["base_seed"], SEED)
        # A run already on the canonical set has nothing to extend.
        self.assertEqual(extension_specs(SEED, 6, 6), ())
        for bad in ((3, 5), (6, 3), (0, 6), (3, 12), (7, 7)):
            with self.assertRaises(ValueError):
                extension_specs(SEED, *bad)

    def test_curator_accepts_only_the_canonical_extension(self):
        saved = saved_recipes()
        verified = verify_extension(metadata(), recorded_groups=3, saved_recipes=saved)
        self.assertEqual(set(verified), {spec["sample_id"] for spec in extension_specs(SEED, 3, 6)})
        self.assertEqual(verify_extension(None, recorded_groups=3, saved_recipes=saved), {})
        # A short extension, a substituted sample and a saved-set mismatch all fail.
        for mutate in (lambda value: value.__setitem__("extension_cases", 8),
                       lambda value: value["recipes"][0].__setitem__("sample_id", "development-30261026-v0"),
                       lambda value: value.__setitem__("saved_cases", 6)):
            broken = metadata()
            mutate(broken)
            with self.assertRaises(ValueError):
                verify_extension(broken, recorded_groups=3, saved_recipes=saved)
        with self.assertRaises(ValueError):
            verify_extension(metadata(), recorded_groups=6, saved_recipes=saved)
        with self.assertRaises(ValueError):
            verify_extension(metadata(), recorded_groups=3, saved_recipes=[])


if __name__ == "__main__":
    unittest.main()
