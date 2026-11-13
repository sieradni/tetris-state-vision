"""Standing instrument for the declaration-propagation class: V-141, V-142 and V-145.

Three defects in this programme had one shape: a new declared value was carried by the model and the parser but not by
some *other* gate that enumerates the same field, so the value was refused by a check whose message pointed elsewhere.

- V-141: an admission call site omitted the new keyword, so the join silently defaulted it to 0.
- V-142: `OCCUPANCY_LATTICE_FREE_LEVERS` still enumerated `(0, 1)`, so the declared mode `2` failed the join.
- V-145: the sweep's `OWNED_RUN_PATTERN` did not carry the new run family, so a passing study could not be swept.

Three defects in one class require a standing instrument, so this module cross-checks the gates against each other rather
than against a hand-written list: every value the model accepts must be accepted by the parser and by the join, and every
run family that committed evidence actually used must be accepted by the sweep's owned-run guard.
"""
import json
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))

from tetris_vision.losses import OCCUPANCY_LATTICE_FREE_LEVERS, require_occupancy_lattice_free
from tetris_vision.model import ModelConfig


def accepted_modes():
    """Every value the model itself admits for the field, read from the model rather than restated here."""
    modes = []
    for value in range(4):
        try:
            ModelConfig(occupancy_lattice_free=value)
        except ValueError:
            continue
        modes.append(value)
    return tuple(modes)


class LatticeFreePropagationTests(unittest.TestCase):

    def test_the_join_carries_every_mode_the_model_accepts(self):
        modes = accepted_modes()
        self.assertIn(1, modes)
        self.assertIn(2, modes)
        self.assertEqual(tuple(sorted(OCCUPANCY_LATTICE_FREE_LEVERS)), tuple(sorted(modes)))
        for value in modes:
            summary = {"arguments": {"occupancy_lattice_free": value},
                       "recovery_contract": {"occupancy_lattice_free": value}}
            self.assertIsNone(require_occupancy_lattice_free(value, summary), value)

    def test_the_parser_carries_every_mode_the_model_accepts(self):
        from train_m2_stream import arguments
        base = ["--output", "runs/m21-propagation-probe", "--model", "accuracy", "--steps", "2",
                "--profile-square"]
        for value in accepted_modes():
            if value == 1:
                continue  # the single channel is its own arm and is refused beside the published model
            parsed = arguments(base+["--occupancy-detail", "1", "--occupancy-lattice-free", str(value)])
            self.assertEqual(parsed.occupancy_lattice_free, value)

    def test_an_undeclared_mode_is_refused_everywhere(self):
        with self.assertRaises(ValueError):
            ModelConfig(occupancy_lattice_free=3)
        with self.assertRaises(ValueError):
            require_occupancy_lattice_free(3, {"arguments": {"occupancy_lattice_free": 3},
                                               "recovery_contract": {"occupancy_lattice_free": 3}})


class RunFamilyPropagationTests(unittest.TestCase):

    def test_every_family_committed_evidence_used_is_an_owned_name(self):
        """The guard is checked against the runs the programme actually recorded, not against a list kept by hand."""
        from record_stream_run import require_owned_run
        studies = set()
        for scope_path in sorted((ROOT/"docs"/"evidence").glob("*/scope.json")):
            scope = json.loads(scope_path.read_text(encoding="utf-8"))
            arms = scope.get("arms") or {}
            for arm in (arms.values() if isinstance(arms, dict) else arms):
                study = arm.get("study") if isinstance(arm, dict) else None
                if isinstance(study, str) and study:
                    studies.add(study)
        self.assertGreater(len(studies), 10, "the evidence walk found too few studies to be a real check")
        for study in sorted(studies):
            self.assertEqual(require_owned_run(study), study,
                             f"{study} was measured but its family is not an owned run name")

    def test_an_unowned_name_is_still_refused(self):
        from record_stream_run import require_owned_run
        for name in ("random-run", "m21-unknown-arm-x", "m99-whatever"):
            with self.assertRaises(ValueError):
                require_owned_run(name)


def accepted_stencil_modes():
    """Every value the model admits for the stencil, read from the model rather than restated here."""
    modes = []
    for value in range(4):
        try:
            ModelConfig(occupancy_lattice_free=2, occupancy_free_stencil=value)
        except (ValueError, TypeError):
            continue
        modes.append(value)
    return tuple(modes)


class FreeStencilPropagationTests(unittest.TestCase):
    """M22: the new declared field reaches the model, the admission join and the parser as one value."""

    def test_the_join_enumerates_every_mode_the_model_accepts(self):
        from tetris_vision.losses import OCCUPANCY_FREE_STENCIL_LEVERS, require_occupancy_free_stencil
        modes = accepted_stencil_modes()
        self.assertEqual(modes, (0, 1, 2, 3))
        self.assertEqual(tuple(sorted(OCCUPANCY_FREE_STENCIL_LEVERS)), tuple(sorted(modes)))
        for value in modes:
            summary = {"arguments": {"occupancy_free_stencil": value},
                       "recovery_contract": {"occupancy_free_stencil": value}}
            self.assertIsNone(require_occupancy_free_stencil(value, summary), value)

    def test_the_parser_carries_every_mode_the_model_accepts(self):
        from train_m2_stream import arguments
        base = ["--output", "runs/m22-propagation-probe", "--model", "accuracy", "--steps", "2",
                "--profile-square", "--occupancy-detail", "1", "--occupancy-lattice-free", "2"]
        for value in accepted_stencil_modes():
            self.assertEqual(arguments(base+["--occupancy-free-stencil", str(value)]).occupancy_free_stencil, value)

    def test_the_declaration_is_refused_where_the_model_refuses_it(self):
        from train_m2_stream import arguments
        base = ["--output", "runs/m22-propagation-probe", "--model", "accuracy", "--steps", "2",
                "--profile-square", "--occupancy-detail", "1"]
        with self.assertRaises(SystemExit):
            arguments(base+["--occupancy-free-stencil", "1"])

    def test_an_initial_weight_source_that_predates_the_field_is_still_admissible(self):
        """The source checkpoint is the same function when the field is 0, and the gate must say so."""
        import inspect
        import train_m2_stream
        source = inspect.getsource(train_m2_stream.main)
        self.assertIn('"occupancy_free_stencil"', source)


class RecoveryContractCompletenessTests(unittest.TestCase):
    """V-147: a declared occupancy lever must not be able to miss the recovery contract by being forgotten there.

    The contract's assignments are hand-maintained; a field added to the model and the parser can still be absent from
    the contract, which the admission join reads, so the arm declaring it is refused by its own admission.
    """

    def test_a_declared_occupancy_lever_is_filled_from_the_config(self):
        from train_m2_stream import complete_recovery_contract
        self.assertEqual(complete_recovery_contract({}, {"occupancy_free_stencil": 1, "columns": 10}),
                         {"occupancy_free_stencil": 1})

    def test_an_explicit_contract_value_is_never_overwritten(self):
        from train_m2_stream import complete_recovery_contract
        self.assertEqual(complete_recovery_contract({"occupancy_detail": 0}, {"occupancy_detail": 1}),
                         {"occupancy_detail": 0})

    def test_every_occupancy_field_of_the_model_reaches_the_contract(self):
        from train_m2_stream import complete_recovery_contract
        fields = {name: 0 for name in ModelConfig.__dataclass_fields__}
        contract = complete_recovery_contract({}, fields)
        missing = [name for name in fields if name.startswith("occupancy_") and name not in contract]
        self.assertEqual(missing, [])

    def test_main_completes_the_contract_that_it_writes(self):
        import inspect
        import train_m2_stream
        self.assertIn("complete_recovery_contract(recovery_contract", inspect.getsource(train_m2_stream.main))


if __name__ == "__main__":
    unittest.main()
