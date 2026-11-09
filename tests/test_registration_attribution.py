"""The declared registration fallback: pure pairing, invariance and the verdict boundary."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))

from registration_attribution import (COLUMNS, F1_FLOOR, ROWS, attribution_verdict, counts_of,
                                      map_cells, matched_indices, summarize)
from record_stream_run import OWNED_RUN_PATTERN


def target():
    """A readable 10x20 bottom-first lattice in normalized pixels."""
    return {"origin": (0.1, 0.9), "u": (0.08, 0.0), "v": (0.0, -0.042)}


def shifted(dx=0.0, dy=0.0, *, origin=None, u=None, v=None):
    base = target()
    origin = base["origin"] if origin is None else origin
    u = base["u"] if u is None else u
    v = base["v"] if v is None else v
    return {"origin": (origin[0]+dx*u[0]+dy*v[0], origin[1]+dx*u[1]+dy*v[1]), "u": u, "v": v}


def case(mapping, occupied=None, logits=None, known=None, *, cluster=20261026):
    cells = COLUMNS*ROWS
    return {"sample_id": f"training-{cluster}-v0", "mapping": mapping,
            "occupied": occupied or [1 if index % 2 else 0 for index in range(cells)],
            "known": known or [True]*cells,
            "logits": logits or [1.0 if index % 2 else -1.0 for index in range(cells)],
            "offsets": [(0.0, 0.0)]*cells}


class PairingTests(unittest.TestCase):
    def test_identical_lattices_map_each_cell_to_itself(self):
        mapping, offsets = map_cells(target(), target())
        self.assertEqual(mapping, list(range(COLUMNS*ROWS)))
        self.assertTrue(all(abs(column) < 1e-9 and abs(row) < 1e-9 for column, row in offsets))

    def test_a_one_cell_shift_maps_to_the_covered_cell(self):
        mapping, _ = map_cells(shifted(dx=1.0), target())
        for r in range(ROWS):
            for c in range(COLUMNS):
                expected = r*COLUMNS+c+1 if c+1 < COLUMNS else None
                self.assertEqual(mapping[r*COLUMNS+c], expected)

    def test_a_downward_shift_maps_to_the_row_below(self):
        mapping, _ = map_cells(shifted(dy=1.0), target())
        self.assertEqual(mapping[0], 1*COLUMNS+0)
        self.assertIsNone(mapping[(ROWS-1)*COLUMNS+0])

    def test_cells_leaving_the_board_are_excluded(self):
        mapping, _ = map_cells(shifted(dx=COLUMNS+2.0), target())
        self.assertTrue(all(value is None for value in mapping))

    def test_paired_raster_equivalence(self):
        """Different hidden states that yield identical geometry must pair identically."""
        left = {"origin": (0.2, 0.8), "u": (0.07, 0.0), "v": (0.0, -0.038)}
        right = {"origin": (0.2, 0.8), "u": (0.07, 0.0), "v": (0.0, -0.038)}
        self.assertEqual(map_cells(left, target()), map_cells(right, target()))

    def test_a_common_translation_of_both_lattices_pairs_identically(self):
        mapping, _ = map_cells(shifted(dx=2.0, dy=1.0), shifted(dx=2.0, dy=1.0))
        self.assertEqual(mapping, list(range(COLUMNS*ROWS)))

    def test_matched_cells_are_the_mapped_and_known_ones(self):
        mapping, _ = map_cells(shifted(dx=1.0), target())
        known = [True]*(COLUMNS*ROWS)
        known[0] = False
        selected = matched_indices(case(mapping, known=known))
        self.assertEqual(len(selected), COLUMNS*ROWS-1-ROWS)
        self.assertNotIn(0, selected)
        self.assertNotIn(COLUMNS-1, selected)


class ArithmeticTests(unittest.TestCase):
    def test_counts_and_the_identity(self):
        counts = counts_of([(1.0, 1), (1.0, 0), (-1.0, 1), (-1.0, 0)])
        self.assertEqual(counts, {"true_positive": 1, "false_positive": 1,
                                  "false_negative": 1, "true_negative": 1})

    def test_a_pure_alignment_error_is_fully_recovered(self):
        """Labels one cell to the right of the model's own index: oracle pairing recovers them."""
        cells = COLUMNS*ROWS
        occupied = [1 if index % 2 else 0 for index in range(cells)]
        mapping = [(index//COLUMNS)*COLUMNS+(index % COLUMNS+1)
                   if (index % COLUMNS) < COLUMNS-1 else None for index in range(cells)]
        logits = [1.0 if mapping[i] is not None and occupied[mapping[i]] else -1.0 for i in range(cells)]
        as_is = case(mapping, occupied=occupied, logits=logits)
        measured = summarize([as_is, case(mapping, occupied=occupied, logits=logits, cluster=20261027)])
        self.assertGreater(measured["oracle"]["f1"], measured["as_is"]["f1"])
        self.assertEqual(measured["mapped_share"], (COLUMNS-1)/COLUMNS)

    def test_empty_boards_do_not_enter_the_deciding_population(self):
        mapping, _ = map_cells(target(), target())
        occupied = [1 if index % 2 else 0 for index in range(COLUMNS*ROWS)]
        logits = [1.0 if label else -1.0 for label in occupied]
        nonempty = case(mapping, occupied=occupied, logits=logits)
        second = case(mapping, occupied=occupied, logits=logits, cluster=20261027)
        empty = case(mapping, occupied=[0]*(COLUMNS*ROWS), logits=[1.0]*(COLUMNS*ROWS), cluster=20261028)
        measured = summarize([nonempty, second, empty])
        self.assertEqual((measured["images"], measured["empty_images"]), (2, 1))
        self.assertEqual(measured["cells"], 2*COLUMNS*ROWS)
        self.assertEqual(measured["as_is_full"]["f1"], 1.0)

    def test_the_ownership_family_and_empty_board_rule(self):
        mapping, _ = map_cells(target(), target())
        with self.assertRaises(ValueError):
            summarize([case(mapping, occupied=[0]*(COLUMNS*ROWS))])

    def test_verdict_needs_both_the_floor_and_an_interval_above_zero(self):
        summary = {"f1_delta": .08, "attribution": {"bootstrap": {"f1_delta": {"low": .02}}}}
        self.assertTrue(attribution_verdict(summary)["registration_implicated"])
        summary = {"f1_delta": .08, "attribution": {"bootstrap": {"f1_delta": {"low": -.01}}}}
        self.assertFalse(attribution_verdict(summary)["registration_implicated"])
        summary = {"f1_delta": .049, "attribution": {"bootstrap": {"f1_delta": {"low": .04}}}}
        self.assertFalse(attribution_verdict(summary)["registration_implicated"])
        summary = {"f1_delta": F1_FLOOR, "attribution": {"bootstrap": {"f1_delta": {"low": .001}}}}
        self.assertTrue(attribution_verdict(summary)["registration_implicated"])

    def test_the_owned_family_is_declared(self):
        for name in ("m3-occscale-registration-r1",):
            self.assertTrue(OWNED_RUN_PATTERN.fullmatch(name), name)


if __name__ == "__main__":
    unittest.main()
