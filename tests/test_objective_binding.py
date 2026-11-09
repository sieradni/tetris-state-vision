"""A run's objective is replayed only where its frozen archive can execute it."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))

from probe_stream_errors import objective_kwargs


class ObjectiveBindingTests(unittest.TestCase):
    def test_a_full_archive_receives_every_recorded_term(self):
        parameters = {"identity_factor", "endpoint_term", "occupancy_factor", "positive_weight"}
        self.assertEqual(objective_kwargs(parameters, identity_factor=4, endpoint_term="length",
                                          occupancy_factor=4, positive_weight=4),
                         {"identity_factor": 4, "endpoint_term": "length",
                          "occupancy_factor": 4, "positive_weight": 4})

    def test_a_pre_term_archive_replays_its_recorded_defaults_by_omission(self):
        parameters = {"identity_factor", "endpoint_term"}
        self.assertEqual(objective_kwargs(parameters, identity_factor=4, endpoint_term="length"),
                         {"identity_factor": 4, "endpoint_term": "length"})

    def test_a_non_default_term_the_archive_cannot_execute_is_refused(self):
        identity_only = {"identity_factor"}
        for kwargs in ({"occupancy_factor": 4}, {"positive_weight": 4}, {"endpoint_term": "separate"}):
            with self.assertRaises(ValueError):
                objective_kwargs(identity_only, **kwargs)
        # A term the archive does have is bound, never refused.
        self.assertEqual(objective_kwargs(identity_only | {"endpoint_term"}, endpoint_term="separate"),
                         {"identity_factor": 1, "endpoint_term": "separate"})

    def test_the_full_archive_is_unchanged_by_the_binding(self):
        parameters = {"identity_factor", "endpoint_term", "occupancy_factor", "positive_weight"}
        self.assertEqual(objective_kwargs(parameters), {"identity_factor": 1, "endpoint_term": "joint",
                                                        "occupancy_factor": 1, "positive_weight": 1})


if __name__ == "__main__":
    unittest.main()
