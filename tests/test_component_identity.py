"""Declared component-identity loss factor: scale-only, never a silent change."""
import importlib.util
import io
from contextlib import redirect_stderr
from pathlib import Path
import sys
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from torch.nn import functional as F
    from tetris_vision.components import component_loss, component_tensors
    from tetris_vision.state import FallingPiece, Icon, Queue, VisionState
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
    from train_m2_stream import arguments


def state(queue, hold):
    return VisionState(None, FallingPiece("unknown"), FallingPiece("unknown"), queue, hold, "needs_review")


def perfect_logits(target):
    def categorical(values, classes):
        return F.one_hot(values.clamp_min(0), classes).float()*20-10
    return {"queue_tokens": categorical(target["queue_tokens"], 8),
            "queue_slot_status": categorical(target["queue_slot_status"], 3),
            "queue_presence": target["queue_presence"].float()*20-10,
            "queue_status": categorical(target["queue_status"], 4),
            "queue_anchors": torch.stack((target["queue_start_observed"], target["queue_end_observed"]), -1).float()*20-10,
            "hold_status": categorical(target["hold_status"], 5), "hold_piece": categorical(target["hold_piece"], 7)}


def fixture(queue=(Icon("observed", "T"), Icon("observed", "I")), hold=Icon("observed", "O")):
    target = component_tensors([state(Queue(queue), hold)], capacity=4)
    return perfect_logits(target), target


@unittest.skipUnless(HAS_TORCH, "install the optional train dependency for M3 gates")
class ComponentIdentityFactorTests(unittest.TestCase):
    def test_factor_one_is_the_published_component_objective(self):
        output, target = fixture()
        baseline, baseline_terms = component_loss(output, target)
        explicit, explicit_terms = component_loss(output, target, identity_factor=1)
        self.assertTrue(torch.equal(baseline, explicit))
        self.assertEqual(set(baseline_terms), set(explicit_terms))
        self.assertTrue(all(torch.equal(baseline_terms[key], explicit_terms[key]) for key in baseline_terms))

    def test_factor_four_scales_only_the_two_identity_terms(self):
        output, target = fixture()
        baseline, terms = component_loss(output, target)
        scaled, scaled_terms = component_loss(output, target, identity_factor=4)
        self.assertTrue(all(torch.equal(terms[key], scaled_terms[key]) for key in terms))
        identity = terms["queue_tokens"]+terms["hold_piece"]
        self.assertTrue(torch.allclose(scaled, baseline+3*identity, atol=1e-6))

    def test_the_factor_never_reweights_masked_or_unknown_identity(self):
        output, target = fixture(queue=(Icon("observed", "T"), Icon("unknown"), Icon("observed", "I")))
        baseline, terms = component_loss(output, target)
        scaled, _ = component_loss(output, target, identity_factor=4)
        identity = terms["queue_tokens"]+terms["hold_piece"]
        self.assertTrue(torch.allclose(scaled, baseline+3*identity, atol=1e-6))
        self.assertEqual(int(target["queue_token_known"].sum()), 3)

    def test_unpublished_factors_are_refused(self):
        output, target = fixture()
        for factor in (0, 2, 3, 8, True, 4.0):
            with self.assertRaisesRegex(ValueError, "component identity factor"):
                component_loss(output, target, identity_factor=factor)

    def test_the_driver_requires_the_joint_heads_for_a_factor(self):
        base = ["--output", "unused", "--model", "accuracy", "--steps", "2", "--profile-square"]
        self.assertEqual(arguments(base).component_identity_factor, 1)
        self.assertEqual(arguments([*base, "--component-capacity", "64", "--component-identity-factor", "4"]
                                   ).component_identity_factor, 4)
        for extra in (["--component-identity-factor", "4"],
                      ["--component-capacity", "64", "--component-identity-factor", "2"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                arguments([*base, *extra])


    def test_shared_identity_readout_reuses_the_cell_type_head(self):
        from dataclasses import replace
        from tetris_vision.model import ModelConfig, TetrisStateNet
        model = TetrisStateNet(replace(ModelConfig.pilot(), component_capacity=4,
                                       component_identity_readout="shared")).eval()
        self.assertIs(model.components.shared_type, model.semantic)
        self.assertIsNotNone(model.components.queue_eos)
        self.assertIsNone(model.components.queue_tokens)
        self.assertIsNone(model.components.hold_piece)
        keys = set(model.state_dict())
        self.assertFalse(any("queue_tokens" in key or "hold_piece" in key for key in keys))
        with torch.no_grad():
            final = model(torch.rand(1, 3, 96, 112), torch.ones(1, 1, 96, 112, dtype=torch.bool))["final"]["components"]
        self.assertEqual(final["queue_tokens"].shape, (1, 5, 8))
        self.assertEqual(final["hold_piece"].shape, (1, 7))

    def test_separate_readout_keeps_the_published_identity_heads(self):
        from dataclasses import replace
        from tetris_vision.model import ModelConfig, TetrisStateNet
        model = TetrisStateNet(replace(ModelConfig.pilot(), component_capacity=4)).eval()
        self.assertIsNone(model.components.shared_type)
        self.assertIn("components.queue_tokens.weight", model.state_dict())
        self.assertFalse(any("shared_type" in key for key in model.state_dict()))
        self.assertEqual(ModelConfig.pilot().component_identity_readout, "separate")

    def test_identity_readout_mode_is_validated(self):
        from dataclasses import replace
        from tetris_vision.model import ModelConfig
        with self.assertRaises(ValueError):
            replace(ModelConfig.pilot(), component_identity_readout="shared")
        with self.assertRaises(ValueError):
            replace(ModelConfig.pilot(), component_capacity=4, component_identity_readout="tied")

    def test_the_driver_declares_the_readout_mode(self):
        base = ["--output", "unused", "--model", "accuracy", "--steps", "2", "--profile-square"]
        self.assertEqual(arguments(base).component_identity_readout, "separate")
        self.assertEqual(arguments([*base, "--component-capacity", "64", "--component-identity-readout", "shared"]
                                   ).component_identity_readout, "shared")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            arguments([*base, "--component-identity-readout", "shared"])

    def test_prototype_readout_conditions_every_class_on_the_image(self):
        from tetris_vision.components import ComponentDecoder
        decoder = ComponentDecoder(32, 4).use_prototype_readout()
        self.assertEqual(tuple(decoder.piece_queries.shape), (1, 8, 32))
        self.assertEqual(tuple(decoder.hold_queries.shape), (1, 4, 32))
        self.assertIsNone(decoder.queue_tokens)
        self.assertIsNone(decoder.hold_piece)
        valid = torch.ones(1, 5, dtype=torch.bool)
        board = torch.zeros(1, 1, 32)
        torch.manual_seed(3)
        with torch.no_grad():
            first = decoder(torch.randn(1, 5, 32), valid, board, capacity=4)
            second = decoder(torch.randn(1, 5, 32), valid, board, capacity=4)
        self.assertEqual(tuple(first["queue_tokens"].shape), (1, 5, 8))
        self.assertEqual(tuple(first["hold_piece"].shape), (1, 7))
        # The class vectors are read from the frame, so the same queries see different classes.
        self.assertFalse(torch.allclose(first["queue_tokens"], second["queue_tokens"]))
        self.assertFalse(torch.allclose(first["hold_piece"], second["hold_piece"]))

    def test_positional_readout_gives_every_position_its_own_image_query(self):
        from dataclasses import replace
        from tetris_vision.components import ComponentDecoder
        from tetris_vision.model import ModelConfig, TetrisStateNet
        decoder = ComponentDecoder(32, 4).use_positional_queries()
        self.assertEqual(tuple(decoder.position_queries.shape), (1, 65, 32))
        self.assertEqual(tuple(decoder.piece_queries.shape), (1, 8, 32))
        self.assertEqual(tuple(decoder.hold_queries.shape), (1, 4, 32))
        self.assertIsNone(decoder.queue_tokens)
        self.assertIsNone(decoder.hold_piece)
        valid = torch.ones(1, 5, dtype=torch.bool)
        board = torch.zeros(1, 1, 32)
        torch.manual_seed(5)
        with torch.no_grad():
            first = decoder(torch.randn(1, 5, 32), valid, board, capacity=4)["queue_tokens"]
            second = decoder(torch.randn(1, 5, 32), valid, board, capacity=4)["queue_tokens"]
        self.assertEqual(tuple(first.shape), (1, 5, 8))
        # The position axis reaches the logits: two positions of one frame differ, and the
        # same position reads a different frame differently.
        self.assertFalse(torch.allclose(first[0, 0], first[0, 1]))
        self.assertFalse(torch.allclose(first, second))
        model = TetrisStateNet(replace(ModelConfig.pilot(), component_capacity=4,
                                       component_identity_readout="positional")).eval()
        self.assertIn("components.position_queries", model.state_dict())

    def test_separate_endpoint_term_takes_the_endpoint_out_of_the_piece_softmax(self):
        from dataclasses import replace
        from tetris_vision.components import ComponentDecoder
        from tetris_vision.model import ModelConfig, TetrisStateNet
        decoder = ComponentDecoder(32, 4).use_positional_queries().use_factorized_endpoint()
        self.assertTrue(decoder.factorized_endpoint)
        valid = torch.ones(1, 5, dtype=torch.bool)
        torch.manual_seed(13)
        with torch.no_grad():
            output = decoder(torch.randn(1, 5, 32), valid, torch.zeros(1, 1, 32), capacity=4)
        self.assertEqual(tuple(output["queue_endpoint"].shape), (1, 5))
        # Emitted columns stay log-probabilities of the factorization, so the ordinary
        # argmax metrics and the position-spread diagnostic keep their meaning.
        self.assertTrue(torch.allclose(output["queue_tokens"].exp().sum(-1), torch.ones(1, 5), atol=1e-5))
        self.assertTrue(torch.allclose(output["queue_tokens"][..., 7],
                                       torch.nn.functional.logsigmoid(output["queue_endpoint"]), atol=1e-6))
        model = TetrisStateNet(replace(ModelConfig.pilot(), component_capacity=4,
                                       component_identity_readout="positional",
                                       component_endpoint_term="separate")).eval()
        self.assertTrue(model.components.factorized_endpoint)
        with self.assertRaises(ValueError):
            replace(ModelConfig.pilot(), component_capacity=4, component_endpoint_term="separate")

    def test_the_separate_endpoint_term_charges_a_constant_piece_class(self):
        from tetris_vision.components import component_loss
        target = {"capacity": 4,
                  "queue_tokens": torch.tensor([[2, 3, 7, -1, -1]]),
                  "queue_token_known": torch.tensor([[True, True, True, False, False]]),
                  "queue_slot_status": torch.tensor([[0, 0, 0, -1]]),
                  "queue_slot_status_known": torch.tensor([[True, True, True, False]]),
                  "queue_presence": torch.tensor([[True, True, False, False]]),
                  "queue_presence_known": torch.tensor([[True, True, True, True]]),
                  "queue_status": torch.tensor([0]), "hold_status": torch.tensor([0]),
                  "hold_piece": torch.tensor([0]), "hold_piece_known": torch.tensor([True]),
                  "queue_start_observed": torch.tensor([True]), "queue_end_observed": torch.tensor([True])}
        logits = torch.zeros(1, 5, 8)
        logits[..., 2] = 8.
        output = {"queue_endpoint": torch.zeros(1, 5), "queue_tokens": logits,
                  "queue_presence": torch.zeros(1, 4), "queue_slot_status": torch.zeros(1, 4, 3),
                  "queue_status": torch.zeros(1, 4), "queue_anchors": torch.zeros(1, 2),
                  "hold_status": torch.zeros(1, 5), "hold_piece": torch.zeros(1, 7)}
        loss, terms = component_loss(output, target, identity_factor=4, endpoint_term="separate")
        self.assertIn("queue_endpoint", terms)
        self.assertGreater(float(terms["queue_tokens"].detach()), .5)
        self.assertGreater(float(loss.detach()), 0)
        with self.assertRaisesRegex(ValueError, "joint endpoint term"):
            component_loss(output, target, endpoint_term="joint")
        del output["queue_endpoint"]
        with self.assertRaisesRegex(ValueError, "separate endpoint term"):
            component_loss(output, target, endpoint_term="separate")

    def test_the_driver_declares_the_endpoint_term(self):
        base = ["--output", "unused", "--model", "accuracy", "--steps", "2", "--profile-square"]
        self.assertEqual(arguments(base).component_endpoint_term, "joint")
        self.assertEqual(arguments([*base, "--component-capacity", "64", "--component-identity-readout",
                                    "positional", "--component-endpoint-term", "separate"]).component_endpoint_term,
                         "separate")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            arguments([*base, "--component-endpoint-term", "separate"])

    def test_the_length_head_replaces_the_per_position_endpoint_question(self):
        from dataclasses import replace
        from tetris_vision.components import ComponentDecoder
        from tetris_vision.model import ModelConfig, TetrisStateNet
        decoder = ComponentDecoder(32, 4).use_positional_queries().use_length_distribution()
        self.assertIsNotNone(decoder.length_direction)
        valid = torch.ones(1, 5, dtype=torch.bool)
        torch.manual_seed(17)
        with torch.no_grad():
            output = decoder(torch.randn(1, 5, 32), valid, torch.zeros(1, 1, 32), capacity=4)
        self.assertNotIn("queue_endpoint", output)
        self.assertEqual(tuple(output["queue_length"].shape), (1, 5))
        # One distribution over lengths0..capacity, and every emitted position stays a
        # proper class distribution so the ordinary metrics keep their meaning.
        self.assertTrue(torch.allclose(output["queue_length"].exp().sum(-1), torch.ones(1), atol=1e-5))
        self.assertTrue(torch.allclose(output["queue_tokens"].exp().sum(-1), torch.ones(1, 5), atol=1e-5))
        model = TetrisStateNet(replace(ModelConfig.pilot(), component_capacity=4,
                                       component_identity_readout="positional",
                                       component_endpoint_term="length")).eval()
        self.assertIsNotNone(model.components.length_direction)
        with self.assertRaises(ValueError):
            replace(ModelConfig.pilot(), component_capacity=4, component_identity_readout="prototype",
                    component_endpoint_term="length")

    def test_the_length_objective_charges_one_decision_and_no_position_endpoint(self):
        from tetris_vision.components import component_loss
        target = {"capacity": 4,
                  "queue_tokens": torch.tensor([[2, 3, 7, -1, -1]]),
                  "queue_token_known": torch.tensor([[True, True, True, False, False]]),
                  "queue_slot_status": torch.tensor([[0, 0, 0, -1]]),
                  "queue_slot_status_known": torch.tensor([[True, True, True, False]]),
                  "queue_presence": torch.tensor([[True, True, False, False]]),
                  "queue_presence_known": torch.tensor([[True, True, True, True]]),
                  "queue_status": torch.tensor([0]), "hold_status": torch.tensor([0]),
                  "hold_piece": torch.tensor([0]), "hold_piece_known": torch.tensor([True]),
                  "queue_start_observed": torch.tensor([True]), "queue_end_observed": torch.tensor([True])}
        logits = torch.zeros(1, 5, 8)
        logits[..., 2] = 8.
        output = {"queue_length": torch.full((1, 5), float(-torch.log(torch.tensor(5.)))),
                  "queue_tokens": logits, "queue_presence": torch.zeros(1, 4),
                  "queue_slot_status": torch.zeros(1, 4, 3), "queue_status": torch.zeros(1, 4),
                  "queue_anchors": torch.zeros(1, 2), "hold_status": torch.zeros(1, 5),
                  "hold_piece": torch.zeros(1, 7)}
        loss, terms = component_loss(output, target, identity_factor=4, endpoint_term="length")
        self.assertIn("queue_length", terms)
        self.assertNotIn("queue_endpoint", terms)
        self.assertGreater(float(terms["queue_length"].detach()), 0)
        self.assertGreater(float(terms["queue_tokens"].detach()), .5)
        self.assertGreater(float(loss.detach()), 0)
        with self.assertRaisesRegex(ValueError, "joint endpoint term"):
            component_loss(output, target, endpoint_term="joint")
        output["queue_endpoint"] = torch.zeros(1, 5)
        with self.assertRaisesRegex(ValueError, "length objective cannot consume"):
            component_loss(output, target, endpoint_term="length")
        del output["queue_endpoint"], output["queue_length"]
        with self.assertRaisesRegex(ValueError, "length objective needs"):
            component_loss(output, target, endpoint_term="length")

    def test_the_driver_declares_the_length_objective(self):
        base = ["--output", "unused", "--model", "accuracy", "--steps", "2", "--profile-square"]
        self.assertEqual(arguments([*base, "--component-capacity", "64", "--component-identity-readout",
                                    "positional", "--component-endpoint-term", "length"]).component_endpoint_term,
                         "length")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            arguments([*base, "--component-capacity", "64", "--component-identity-readout", "prototype",
                       "--component-endpoint-term", "length"])

    def test_every_declared_endpoint_objective_is_accepted_by_the_driver_and_the_gate(self):
        from tetris_vision.components import ENDPOINT_TERMS, require_endpoint_term
        from train_m2_stream import ENDPOINT_TERMS as DRIVER_TERMS
        self.assertEqual(tuple(DRIVER_TERMS), tuple(ENDPOINT_TERMS))
        base = ["--output", "unused", "--model", "accuracy", "--steps", "2", "--profile-square"]
        for mode in ENDPOINT_TERMS:
            require_endpoint_term(mode, {"arguments": {"component_endpoint_term": mode},
                                         "recovery_contract": {"component_endpoint_term": mode}})
            if mode != "joint":
                self.assertEqual(arguments([*base, "--component-capacity", "64", "--component-identity-readout",
                                            "positional", "--component-endpoint-term", mode]).component_endpoint_term,
                                 mode)
        with self.assertRaisesRegex(ValueError, "endpoint term differs"):
            require_endpoint_term("length", {"arguments": {"component_endpoint_term": "separate"},
                                            "recovery_contract": {"component_endpoint_term": "separate"}})
        with self.assertRaisesRegex(ValueError, "endpoint term differs"):
            require_endpoint_term("queue", {"arguments": {}, "recovery_contract": {}})

    def test_the_occupancy_factor_is_declared_once_and_gated_consistently(self):
        from tetris_vision.losses import OCCUPANCY_FACTORS, require_occupancy_factor
        from train_m2_stream import OCCUPANCY_FACTORS as DRIVER_FACTORS
        self.assertEqual(tuple(DRIVER_FACTORS), tuple(OCCUPANCY_FACTORS))
        self.assertEqual(tuple(OCCUPANCY_FACTORS), (1, 4))
        for factor in OCCUPANCY_FACTORS:
            require_occupancy_factor(factor, {"arguments": {"occupancy_factor": factor},
                                              "recovery_contract": {"occupancy_factor": factor}})
        base = ["--output", "unused", "--model", "accuracy", "--steps", "2", "--profile-square"]
        self.assertEqual(arguments([*base, "--component-capacity", "64", "--component-identity-readout",
                                    "positional", "--occupancy-factor", "4"]).occupancy_factor, 4)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            arguments([*base, "--component-capacity", "64", "--component-identity-readout",
                       "positional", "--occupancy-factor", "2"])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            arguments([*base, "--occupancy-factor", "4"])
        with self.assertRaisesRegex(ValueError, "occupancy factor differs"):
            require_occupancy_factor(4, {"arguments": {"occupancy_factor": 1},
                                         "recovery_contract": {"occupancy_factor": 1}})

    def test_the_curator_reads_the_canonical_native_sidecars(self):
        # The replay receipt is `<name>-job.json` but its captured stdout/stderr are
        # `<name>.log` and `<name>.stderr.log`; the copy loop must read exactly these.
        from record_stream_errors import NATIVE_SIDECARS
        self.assertEqual(NATIVE_SIDECARS, ("-job.json", ".log", ".stderr.log"))

    def test_the_fit_run_family_is_owned(self):
        from record_stream_run import require_owned_run
        self.assertEqual(require_owned_run("m3-fit-occupancy-r1"), "m3-fit-occupancy-r1")
        for name in ("fit-occupancy-r1", "m3-fit_occupancy-r1", "m3-fit"):
            with self.assertRaises(ValueError):
                require_owned_run(name)

    def test_positional_readout_is_validated_and_declared_by_the_driver(self):
        from dataclasses import replace
        from tetris_vision.model import ModelConfig
        with self.assertRaises(ValueError):
            replace(ModelConfig.pilot(), component_identity_readout="positional")
        base = ["--output", "unused", "--model", "accuracy", "--steps", "2", "--profile-square"]
        self.assertEqual(arguments([*base, "--component-capacity", "64",
                                    "--component-identity-readout", "positional"]).component_identity_readout,
                         "positional")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            arguments([*base, "--component-identity-readout", "positional"])

    def test_prototype_readout_is_exclusive_and_validated(self):
        from dataclasses import replace
        from tetris_vision.components import ComponentDecoder
        from tetris_vision.model import ModelConfig, TetrisStateNet
        with self.assertRaisesRegex(ValueError, "exclusive"):
            ComponentDecoder(32, 4).use_prototype_readout().use_prototype_readout()
        with self.assertRaises(ValueError):
            replace(ModelConfig.pilot(), component_identity_readout="prototype")
        model = TetrisStateNet(replace(ModelConfig.pilot(), component_capacity=4,
                                       component_identity_readout="prototype")).eval()
        keys = set(model.state_dict())
        self.assertIn("components.piece_queries", keys)
        self.assertIn("components.hold_queries", keys)
        self.assertFalse(any("queue_tokens" in key or "hold_piece" in key or "shared_type" in key
                             for key in keys))

    def test_the_driver_declares_the_prototype_readout(self):
        base = ["--output", "unused", "--model", "accuracy", "--steps", "2", "--profile-square"]
        self.assertEqual(arguments([*base, "--component-capacity", "64",
                                    "--component-identity-readout", "prototype"]).component_identity_readout,
                         "prototype")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            arguments([*base, "--component-identity-readout", "prototype"])

    def test_the_declared_lever_family_is_owned_by_the_replay_guard(self):
        from record_stream_run import require_owned_run
        for name in ("m3-identity-factor-r1", "m3-identity-factor-r1-errors-r2", "m3-stream-joint-fit-r1"):
            self.assertEqual(require_owned_run(name), name)
        for name in ("../m3-identity-factor-r1", "m3-identity-factor-r1/../x", "identity-factor-r1", "m3-identity-factor-r1 "):
            with self.assertRaises(ValueError):
                require_owned_run(name)


if __name__ == "__main__":
    unittest.main()
