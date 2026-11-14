"""Controls for the M32 distribution-shaping term.

`docs/M32_MASS_MATCH.md` declares the term as supervision of the read's mass, not of any boundary or
cell. These tests bind what that means: the term is zero exactly when a board's predicted mass equals
its occupancy, it rises monotonically as the mass moves away, it ignores `unknown` cells, and it adds
no parameter or inference-time quantity.
"""
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

from tetris_vision.losses import (MASS_WEIGHT, OCCUPANCY_MASS_ARMS, mass_match_term,  # noqa: E402
                                  require_occupancy_mass)

VISIBLE, SPAWN, COLUMNS = 20, 6, 10


def board(logit):
    return torch.cat((torch.full((1, VISIBLE, COLUMNS), float(logit)),
                      torch.zeros(1, SPAWN, COLUMNS)), 1)


def target(occupied_rows):
    occupied = torch.zeros(1, VISIBLE, COLUMNS)
    occupied[0, :occupied_rows] = 1.0
    known = torch.ones(1, VISIBLE, COLUMNS, dtype=torch.bool)
    return {"occupied": occupied, "known": known}


class TermTest(unittest.TestCase):
    def test_the_term_is_zero_when_the_mass_matches(self):
        # 4 of 200 rows occupied is 2% of the board; a logit chosen to put 2% of mass above zero
        fraction = 4/VISIBLE
        logit = float(torch.logit(torch.tensor(fraction)))
        value = mass_match_term({"occupancy": board(logit)}, target(4))
        self.assertLess(float(value), 1e-4)

    def test_the_term_grows_as_the_mass_moves_away(self):
        values = [float(mass_match_term({"occupancy": board(logit)}, target(4)))
                  for logit in (-6.0, -1.0, 0.0, 3.0)]
        self.assertLess(values[1], values[2])
        self.assertLess(values[2], values[3])
        self.assertGreater(values[0], 0.0)

    def test_unknown_cells_are_never_counted_as_empty(self):
        base = target(4)
        known = base["known"].clone()
        known[0, :] = False
        value = mass_match_term({"occupancy": board(0.0)},
                                {"occupied": base["occupied"], "known": known})
        self.assertEqual(float(value), 0.0)

    def test_the_term_is_pure_supervision(self):
        self.assertEqual(MASS_WEIGHT, 1.0)
        self.assertIn(1, OCCUPANCY_MASS_ARMS)
        with self.assertRaises(ValueError):
            require_occupancy_mass(1, {"arguments": {}, "recovery_contract": {}})
        require_occupancy_mass(1, {"arguments": {"occupancy_mass": 1},
                                   "recovery_contract": {"occupancy_mass": 1}})


if __name__ == "__main__":
    unittest.main()
