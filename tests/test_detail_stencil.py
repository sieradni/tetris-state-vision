"""The declared 2x2 in-cell stencil: same tap, four quadrant samples, identity at zero."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))

import torch
from tetris_vision.losses import require_occupancy_detail_stencil
from tetris_vision.model import ModelConfig, TetrisStateNet

from occupancy_sweep import sweep_verdict
from record_precision_lever import mechanism, paired_bootstrap, precision_of
from record_representation_lever import arm_summary, lever_verdict
import record_stencil_lever as curator
from record_stream_run import OWNED_RUN_PATTERN


def pilot(**overrides):
    return ModelConfig(**{**ModelConfig.pilot().__dict__, **overrides})


def inputs(seed=20261026, size=32):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, size, size, generator=generator), torch.ones(1, 1, size, size, dtype=torch.bool)


def forward(config, seed=20261026):
    torch.manual_seed(seed)
    model = TetrisStateNet(config).eval()
    pixels, valid = inputs()
    with torch.no_grad():
        return model, model(pixels, valid)["final"]["occupancy"]


class DetailStencilTests(unittest.TestCase):
    def test_the_default_is_the_published_configuration(self):
        self.assertEqual(ModelConfig().occupancy_detail_stencil, 0)
        self.assertEqual(ModelConfig.pilot().occupancy_detail_stencil, 0)

    def test_an_undeclared_value_is_refused(self):
        for value in (2, -1, True, 1.0):
            with self.assertRaises(ValueError):
                ModelConfig(occupancy_detail=1, occupancy_detail_stencil=value)

    def test_the_stencil_is_a_form_of_the_detail_tap_and_needs_it(self):
        with self.assertRaises(ValueError):
            ModelConfig(occupancy_detail=0, occupancy_detail_stencil=1)
        self.assertIsNone(forward(pilot(occupancy_detail=0))[0].occupancy_detail_tap)

    def test_the_stencil_reads_four_samples_through_the_same_projection(self):
        model, _ = forward(pilot(occupancy_detail=1, occupancy_detail_stencil=1))
        self.assertIsNotNone(model.occupancy_detail_tap)
        self.assertEqual(model.occupancy_detail_tap.in_features, 4*pilot().widths[0])
        self.assertEqual(tuple(model.detail_stencil.shape), (1, 1, 4, 2))
        self.assertEqual(sorted(map(tuple, model.detail_stencil.reshape(4, 2).tolist())),
                         sorted(map(tuple, [[-.25, -.25], [.25, -.25], [-.25, .25], [.25, .25]])))
        single, _ = forward(pilot(occupancy_detail=1))
        self.assertEqual(single.occupancy_detail_tap.in_features, pilot().widths[0])
        for model in (single, model):
            self.assertNotIn("detail_stencil", model.state_dict())

    def test_each_arm_is_exactly_the_published_function_at_step_zero(self):
        _, plain = forward(pilot())
        for stencil in (0, 1):
            _, levered = forward(pilot(occupancy_detail=1, occupancy_detail_stencil=stencil))
            self.assertTrue(torch.equal(plain, levered), stencil)

    def test_the_arms_differ_only_in_the_tap_and_share_every_other_tensor(self):
        control, _ = forward(pilot(occupancy_detail=1))
        lever, _ = forward(pilot(occupancy_detail=1, occupancy_detail_stencil=1))
        control_keys, lever_keys = set(control.state_dict()), set(lever.state_dict())
        self.assertEqual(control_keys, lever_keys)
        self.assertNotEqual(tuple(control.state_dict()["occupancy_detail_tap.weight"].shape),
                            tuple(lever.state_dict()["occupancy_detail_tap.weight"].shape))
        for name, value in control.state_dict().items():
            if name.startswith("occupancy_detail_tap."):
                continue
            self.assertTrue(torch.equal(value, lever.state_dict()[name]), name)
        self.assertEqual(tuple(lever.state_dict()["occupancy_detail_tap.weight"].shape),
                         (pilot().dimension, 4*pilot().widths[0]))
        self.assertEqual(tuple(control.state_dict()["occupancy_detail_tap.weight"].shape),
                         (pilot().dimension, pilot().widths[0]))

    def test_a_trained_stencil_changes_the_decision(self):
        _, plain = forward(pilot())
        model, untrained = forward(pilot(occupancy_detail=1, occupancy_detail_stencil=1))
        self.assertTrue(torch.equal(plain, untrained))
        with torch.no_grad():
            model.occupancy_detail_tap.weight.fill_(0.25)
        pixels, valid = inputs()
        with torch.no_grad():
            changed = model(pixels, valid)["final"]["occupancy"]
        self.assertFalse(torch.equal(plain, changed))

    def test_relabelling_the_hidden_sample_order_cannot_change_the_decision(self):
        """Paired-raster equivalence: identical pixels, permuted hidden sample order."""
        torch.manual_seed(7)
        lever = TetrisStateNet(pilot(occupancy_detail=1, occupancy_detail_stencil=1)).eval()
        with torch.no_grad():
            lever.occupancy_detail_tap.weight.normal_(0, .5)
            lever.occupancy_detail_tap.bias.normal_(0, .1)
        permuted = deepcopy(lever)
        order = torch.tensor([2, 0, 3, 1])
        width = lever.occupancy_detail_tap.in_features//4
        with torch.no_grad():
            permuted.detail_stencil.copy_(lever.detail_stencil[:, :, order])
            weight = lever.occupancy_detail_tap.weight.view(-1, 4, width)[:, order]
            permuted.occupancy_detail_tap.weight.copy_(weight.reshape(-1, 4*width))
        pixels, valid = inputs()
        with torch.no_grad():
            first = lever(pixels, valid)["final"]["occupancy"]
            second = permuted(pixels, valid)["final"]["occupancy"]
        self.assertFalse(torch.equal(lever.detail_stencil, permuted.detail_stencil))
        self.assertTrue(torch.allclose(first, second, atol=1e-5, rtol=1e-4))

    def test_the_admission_join_accepts_only_a_matching_declaration(self):
        summary = {"arguments": {"occupancy_detail_stencil": 1},
                   "recovery_contract": {"occupancy_detail_stencil": 1}}
        self.assertIsNone(require_occupancy_detail_stencil(1, summary))
        require_occupancy_detail_stencil(0, {"arguments": {}, "recovery_contract": {}})
        for stencil, record in ((1, {"arguments": {"occupancy_detail_stencil": 0},
                                     "recovery_contract": {"occupancy_detail_stencil": 1}}),
                                (1, {"arguments": {"occupancy_detail_stencil": 1},
                                     "recovery_contract": {}}),
                                (2, summary), (True, {"arguments": {"occupancy_detail_stencil": 1},
                                                      "recovery_contract": {"occupancy_detail_stencil": 1}})):
            with self.assertRaises(ValueError):
                require_occupancy_detail_stencil(stencil, record)

    def test_the_owned_run_family_is_declared(self):
        for name in ("m3-occdstencil-lever-r1", "m3-occdstencil-control-r1",
                     "m3-occdstencil-lever-r1-sweep-r1", "m3-occdstencil-control-r1-sweep-profile-r1"):
            self.assertTrue(OWNED_RUN_PATTERN.fullmatch(name), name)
        self.assertFalse(OWNED_RUN_PATTERN.fullmatch("m3-occdstencil-"))


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT/"docs/evidence/m3-occupancy-stencil-r1/scope.json"


class StencilLeverRecordTests(unittest.TestCase):
    """The stencil record re-derives from its committed receipts, and the prose matches them."""

    @classmethod
    def setUpClass(cls):
        cls.record = json.loads(EVIDENCE.read_text(encoding="utf-8"))
        cls.sweeps = {arm: json.loads((ROOT/"runs"/f"m3-occdstencil-{arm}-r1-sweep-r1"/"result.json")
                                      .read_text(encoding="utf-8")) for arm in curator.ARMS}
        cls.studies = {arm: json.loads((ROOT/"runs"/f"m3-occdstencil-{arm}-r1"/"summary.json")
                                       .read_text(encoding="utf-8")) for arm in curator.ARMS}

    def training(self, arm):
        return [row for row in self.sweeps[arm]["cases"] if row["split"] == "training"]

    def test_the_record_is_a_passed_measurement_of_the_declared_instrument(self):
        self.assertEqual(self.record["status"], "passed")
        self.assertEqual(self.record["instrument"], "m3-occupancy-stencil-r1")
        self.assertEqual(self.record["predeclaration"], "docs/M3_OCCUPANCY_STENCIL.md")

    def test_the_population_and_work_arithmetic_are_the_declared_ones(self):
        population = self.record["population"]
        self.assertEqual((population["images"], population["nonempty_boards"]), (36, 28))
        self.assertEqual((population["occupied_cells"], population["empty_cells"]), (667, 4933))
        self.assertEqual(population["contexts"], 12)
        self.assertEqual(self.record["envelope"]["work_items"],
                         {"profile": 2, "study": 64, "sweep_profile": 2, "sweep": 54, "curation": 4})
        for arm, sweep in self.sweeps.items():
            self.assertEqual(sweep["forwards"], 54, arm)
            self.assertEqual(len(self.training(arm)), 36, arm)

    def test_each_arm_re_derives_from_its_own_sweep(self):
        for arm in curator.ARMS:
            measured = arm_summary(self.sweeps[arm]["deciding"])
            measured["precision"] = round(precision_of(measured["operating_counts"]), 6)
            for key in ("auc", "per_cell_f1", "best", "verdict", "precision", "operating_counts"):
                self.assertEqual(self.record["arms"][arm][key], measured[key], (arm, key))

    def test_the_sweeps_use_the_standing_grid_and_re_derive_their_verdict(self):
        for arm, sweep in self.sweeps.items():
            self.assertEqual(len(sweep["deciding"]["thresholds"]), 81, arm)
            self.assertEqual(sweep["training_population"]["evaluated_groups"], 12, arm)
            self.assertEqual(sweep["verdict"], sweep_verdict(sweep["deciding"]), arm)

    def test_the_paired_bootstrap_re_derives_over_twelve_clusters(self):
        expected = paired_bootstrap(self.training("lever"), self.training("control"))
        self.assertEqual(self.record["attribution"]["cluster_bootstrap"], expected)
        self.assertEqual(expected["clusters"], 12)

    def test_the_verdict_and_mechanism_are_the_predeclared_rules(self):
        arms = self.record["arms"]
        verdict = lever_verdict(arms["lever"]["per_cell_f1"], arms["lever"]["verdict"],
                                arms["lever"]["best"]["f1"]-arms["control"]["best"]["f1"])
        self.assertEqual(self.record["verdict"], verdict)
        self.assertEqual(verdict["conditions"],
                         {"per_cell_f1": False, "ranking": False, "attribution": True})
        record = mechanism(arms["lever"]["precision"]-arms["control"]["precision"],
                           self.record["attribution"]["cluster_bootstrap"])
        self.assertEqual(self.record["mechanism"], record)
        self.assertFalse(record["precision_improved"])
        self.assertGreater(record["precision_delta_interval"]["low"], 0)

    def test_the_pair_is_matched_and_differs_only_in_the_stencil(self):
        self.assertEqual(len({self.studies[arm]["catalogue_sha256"] for arm in curator.ARMS}), 1)
        for arm in curator.ARMS:
            self.assertEqual(self.studies[arm]["arguments"]["occupancy_detail"], 1)
            self.assertEqual(self.studies[arm]["arguments"]["occupancy_detail_stencil"], curator.ARMS[arm])
            self.assertEqual(self.studies[arm]["arguments"]["occupancy_context"], 0)
            self.assertEqual((self.studies[arm]["status"], self.studies[arm]["steps"]), ("complete", 64))
        self.assertEqual(self.record["matched"]["config_difference"], ["occupancy_detail_stencil"])
        self.assertFalse(self.record["reference"]["matched_to_this_pass"])

    def test_the_published_gaps_and_next_step_match_the_evidence(self):
        arm = self.record["arms"]["lever"]
        self.assertEqual((round(0.90-arm["auc"], 3), round(0.70-arm["best"]["f1"], 3)), (0.020, 0.209))
        text = (ROOT/"docs/M3_OCCUPANCY_STENCIL.md").read_text(encoding="utf-8")
        self.assertIn("0.020 short of 0.90", text)
        self.assertIn("0.209 short of 0.70", text)
        self.assertIn("precision_improved = false", text)
        for name in ("docs/M3_OCCUPANCY_RANKING.md", "docs/ROADMAP.md", "docs/CONTINUATION_STATE.md"):
            body = (ROOT/name).read_text(encoding="utf-8")
            self.assertIn("encoder resolution", body, name)
            self.assertIn("stencil", body.lower(), name)

if __name__ == "__main__":
    unittest.main()
