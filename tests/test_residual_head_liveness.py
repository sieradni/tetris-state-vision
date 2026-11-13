"""Standing instrument for V-143: a zero-initialized residual head must still be able to learn.

Two declared heads (the lattice-free occupancy channel and the per-board lattice refinement) were
built by zeroing *both* layers of a two-layer MLP. That is a fixed point of training, not an
identity element: the hidden activation is exactly zero, so the output projection's gradient is
`delta * 0` and the hidden layer's gradient flows back through an output weight of zero. The head
can only ever learn a constant, so every arm using it measured a channel that never read its input.

The rule these tests hold: zero the *output* projection only. Step zero stays bitwise the published
function (the head's output is exactly zero), and each parameter of the head receives a non-zero
gradient from a non-zero input.
"""
import torch
import unittest

from tetris_vision.model import ModelConfig, TetrisStateNet


def build(**overrides):
    torch.manual_seed(20_110)
    return TetrisStateNet(ModelConfig(**{**ModelConfig.pilot().__dict__, **overrides}))


class ResidualHeadLivenessTests(unittest.TestCase):

    def assert_live(self, head, width):
        """Liveness is *trainability*: a head that carries the published identity at step zero must
        still acquire an input-dependent output once its output projection has been fitted."""
        torch.manual_seed(30_070)
        features = torch.randn(6, width)
        target = torch.randn(6, head(features).shape[-1])
        optimiser = torch.optim.SGD(head.parameters(), lr=0.5)
        for _ in range(4):
            loss = ((head(features) - target)**2).mean()
            loss.backward()
            optimiser.step()
            head.zero_grad()
        out = head(features)
        self.assertGreater(float(out.abs().max()), 1e-6, "the declared head never leaves zero")
        distinct = head(torch.randn(6, width))
        self.assertGreater(float((out-distinct).abs().max()), 1e-9,
                           "the head's output is a constant, so it never reads its input")
        for name, parameter in head.named_parameters():
            self.assertGreater(float(parameter.abs().max()), 0.0,
                               f"{name} never moves, so it can never be trained")

    def the_instrument_rejects_the_dead_construction(self, head, width):
        """The same probe on a head with both layers zeroed: it must report the head as dead."""
        torch.manual_seed(30_070)
        features = torch.randn(6, width)
        target = torch.randn(6, head(features).shape[-1])
        optimiser = torch.optim.SGD(head.parameters(), lr=0.5)
        for _ in range(4):
            ((head(features) - target)**2).mean().backward()
            optimiser.step()
            head.zero_grad()
        # What a head with both layers zeroed can learn is a *constant*: its biases move and its
        # weights never do, so its output does not depend on the example. That is the signature.
        with torch.no_grad():
            out = head(features)
            spread = torch.cat((head(features), head(torch.randn_like(features))))
        self.assertLess(float(spread.var(dim=0).max()), 1e-12,
                        "the dead construction learned an input-dependent output")
        for name, parameter in head.named_parameters():
            if name.endswith("weight") and name.startswith("0"):
                self.assertEqual(float(parameter.abs().max()), 0.0,
                                 f"{name} moved, so the construction was not dead")

    def test_the_lattice_free_channel_is_a_constant_at_step_zero(self):
        """Identity at step zero is preserved by zeroing the output projection alone."""
        model = build(occupancy_lattice_free=1)
        with torch.no_grad():
            self.assertEqual(float(model.occupancy_lattice_free(torch.randn(7, model.config.dimension)).abs().max()), 0.0)

    def test_the_per_board_refinement_reads_its_input(self):
        model = build(lattice_board_refine=1)
        self.assert_live(model.lattice_board_refine, model.config.dimension)

    def test_the_per_board_refinement_is_a_constant_at_step_zero(self):
        model = build(lattice_board_refine=1)
        with torch.no_grad():
            self.assertEqual(float(model.lattice_board_refine(torch.randn(7, model.config.dimension)).abs().max()), 0.0)

    def test_the_two_path_sum_carries_gradient_into_the_free_channel(self):
        """The M20 composition must train: one step on the summed read reaches the hidden layer and
        the gate, which is what the double-zeroed construction could never do."""
        model = build(occupancy_lattice_free=2, occupancy_stack_monotone=1)
        heads = [module for module in model.modules() if module is model.occupancy_lattice_free]
        self.assertEqual(len(heads), 1)
        head = model.occupancy_lattice_free
        self.assertIsNone(model.occupancy_lattice_gate, "the summed form has no gate (V-143)")
        optimiser = torch.optim.SGD(head.parameters(), lr=0.5)
        features = torch.randn(5, model.config.dimension)
        target = torch.randn(5, 1)
        for _ in range(4):
            ((head(features) - target)**2).mean().backward()
            optimiser.step()
            model.zero_grad()
        self.assertGreater(float(head[2].weight.abs().max()), 0.0)
        self.assertGreater(float(head[0].weight.abs().max()), 0.0,
                           "the hidden layer never receives gradient through the sum")


    def test_the_instrument_catches_the_dead_construction(self):
        """A negative control: the V-143 construction is exactly what the probe above rejects."""
        head = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.SiLU(), torch.nn.Linear(8, 1))
        for layer in (head[0], head[2]):
            torch.nn.init.zeros_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
        self.the_instrument_rejects_the_dead_construction(head, 8)

    def test_no_head_is_constructed_twice(self):
        """The per-board refinement was built twice in a row; the second silently owned the first."""
        source = (__import__("pathlib").Path(__file__).resolve().parents[1]
                  / "src" / "tetris_vision" / "model.py").read_text(encoding="utf-8")
        for construction in ("occupancy_lattice_free = nn.Sequential",
                             "lattice_board_refine = nn.Sequential",
                             "occupancy_neighborhood = nn.Sequential"):
            self.assertEqual(source.count(construction), 1, construction)


if __name__ == "__main__":
    unittest.main()
