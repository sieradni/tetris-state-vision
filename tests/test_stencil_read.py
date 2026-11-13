"""The M22 coarse local stencil: a fixed-lattice detail channel on the free read, and its measured block.

`docs/M22_STENCIL_READ.md` declares six arms and the bar they are judged by. These tests hold the mechanism the block
depends on: that the stencil reads divisions of the capture rather than the decoded lattice - so perturbing the lattice
leaves it unchanged while the lattice-sampled read moves - that it is identity at step zero and learns away from it, that
it is kept as its own part so the pooled channel can be scored exactly, that the declarations reach the real parser, and
that the curated record says what the predeclaration said it would.
"""
import json
import subprocess
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT/"src", ROOT/"scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from tetris_vision.model import CONTRAST_LIMIT, ModelConfig, TetrisStateNet, stencil_contrast

SCOPE = ROOT/"docs/evidence/m23-contrast-r1/scope.json"
DOC = ROOT/"docs/M22_STENCIL_READ.md"
DOC_CONTRAST = ROOT/"docs/M23_CONTRAST_READ.md"
EARLIER = {"control", "envelope", "crossing", "bare", "stencil", "stencil-crossing", "centred", "centred-crossing"}
ARMS = {"control": (0, 0, 0, 0), "envelope": (2, 1, 0, 0), "crossing": (2, 1, 1, 0),
        "bare": (2, 0, 0, 0), "stencil": (2, 1, 0, 1), "stencil-crossing": (2, 1, 1, 1),
        "centred": (2, 1, 0, 2), "centred-crossing": (2, 1, 1, 2),
        "norm": (2, 1, 0, 3), "norm-cross": (2, 1, 1, 3)}


def config(**overrides):
    """Every arm carries the published detail tap; the declared fields are the only difference."""
    return ModelConfig(**{**ModelConfig.pilot().__dict__, "occupancy_detail": 1, **overrides})


def pair(seed=7):
    """The same arm with and without the stencil, built from the same seed."""
    base = {"occupancy_lattice_free": 2, "occupancy_stack_monotone": 1}
    torch.manual_seed(seed)
    with_stencil = TetrisStateNet(config(**base, occupancy_free_stencil=1))
    torch.manual_seed(seed)
    without = TetrisStateNet(config(**base))
    return with_stencil, without


def forward(model, seed=11):
    generator = torch.Generator().manual_seed(seed)
    pixels = torch.rand(2, 3, 96, 96, generator=generator)
    valid = torch.ones(2, 1, 96, 96, dtype=torch.bool)
    with torch.no_grad():
        return model(pixels, valid)["final"]


def fill_detail(model, seed=31):
    """Fill the published detail tap, so the declared sampling geometry is visible at step zero."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in (model.occupancy_detail_tap.weight, model.occupancy_detail_tap.bias):
            parameter.copy_(torch.randn(parameter.shape, generator=generator)*.2)


def fill_stencil(model, seed=5_021):
    """Fill the stencil's output projection from a local generator, so these weights do not depend on RNG order."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in (model.occupancy_free_stencil[2].weight, model.occupancy_free_stencil[2].bias):
            parameter.copy_(torch.randn(parameter.shape, generator=generator)*.25)


class DeclarationTest(unittest.TestCase):
    def test_the_default_is_the_pooled_channel_only(self):
        self.assertEqual(config().occupancy_free_stencil, 0)

    def test_the_stencil_is_declared_only_beside_the_summed_read(self):
        with self.assertRaises(ValueError):
            config(occupancy_lattice_free=1, occupancy_free_stencil=1)
        with self.assertRaises(ValueError):
            config(occupancy_free_stencil=1)
        for value in (0, 1, 2):
            config(occupancy_lattice_free=value, occupancy_free_stencil=0)

    def test_the_stencil_is_a_declared_mode(self):
        for value in (-1, 4, True, 1.0, None):
            with self.assertRaises((ValueError, TypeError)):
                config(occupancy_lattice_free=2, occupancy_free_stencil=value)

    def test_the_parser_carries_it_and_refuses_it_alone(self):
        from train_m2_stream import arguments
        base = ["--output", "runs/m22-test", "--model", "accuracy", "--steps", "2", "--profile-square",
                "--occupancy-detail", "1"]
        parsed = arguments(base+["--occupancy-lattice-free", "2", "--occupancy-free-stencil", "1"])
        self.assertEqual((parsed.occupancy_lattice_free, parsed.occupancy_free_stencil), (2, 1))
        parsed = arguments(base)
        self.assertEqual(parsed.occupancy_free_stencil, 0)
        with self.assertRaises(SystemExit):
            arguments(base+["--occupancy-free-stencil", "1"])
        with self.assertRaises(SystemExit):
            arguments(base+["--occupancy-lattice-free", "2", "--occupancy-free-stencil", "5"])

    def test_the_admission_join_enumerates_both_modes(self):
        from tetris_vision.losses import require_occupancy_free_stencil
        for value in (0, 1):
            summary = {"arguments": {"occupancy_free_stencil": value},
                       "recovery_contract": {"occupancy_free_stencil": value}}
            require_occupancy_free_stencil(value, summary)
        with self.assertRaises(ValueError):
            require_occupancy_free_stencil(1, {"arguments": {"occupancy_free_stencil": 0},
                                               "recovery_contract": {"occupancy_free_stencil": 0}})


class MechanismTest(unittest.TestCase):
    def test_the_stencil_is_identity_at_step_zero(self):
        with_stencil, without = pair()
        a, b = forward(with_stencil), forward(without)
        self.assertEqual(sorted(a["occupancy_parts"]), ["free", "lattice", "local"])
        self.assertEqual(float(a["occupancy_parts"]["local"].abs().max()), 0.0)
        self.assertTrue(torch.equal(a["occupancy"], b["occupancy"]))
        self.assertTrue(torch.equal(a["occupancy_parts"]["free"], b["occupancy_parts"]["free"]))

    def test_the_stencil_changes_the_read_once_it_is_trained(self):
        with_stencil, without = pair()
        fill_stencil(with_stencil)
        a, b = forward(with_stencil), forward(without)
        self.assertFalse(torch.equal(a["occupancy"], b["occupancy"]))
        self.assertGreater(float(a["occupancy_parts"]["local"].abs().max()), 0.0)

    def test_the_local_part_splits_the_free_channel_exactly(self):
        with_stencil, without = pair()
        fill_stencil(with_stencil)
        a, b = forward(with_stencil), forward(without)
        self.assertTrue(torch.allclose(a["occupancy_parts"]["free"],
                                       b["occupancy_parts"]["free"]+a["occupancy_parts"]["local"], atol=1e-5))
        self.assertFalse(torch.allclose(a["occupancy_parts"]["local"], torch.zeros_like(a["occupancy_parts"]["local"])))

    def test_the_stencil_does_not_read_the_decoded_lattice(self):
        """The claim: the stencil is on the fixed lattice, so moving the lattice sample points cannot move it.

        The detail tap is filled first because it is zero-initialized at step zero by design (V-143), so until it is
        trained the declared sampling geometry changes nothing at all and the comparison would be vacuous.
        """
        with_stencil, _ = pair()
        fill_stencil(with_stencil)
        fill_detail(with_stencil)
        before = forward(with_stencil)
        with torch.no_grad():
            with_stencil.tap_offset.add_(torch.tensor([[[[.35, -.3]]]], dtype=with_stencil.tap_offset.dtype))
        after = forward(with_stencil)
        self.assertTrue(torch.equal(before["occupancy_parts"]["local"], after["occupancy_parts"]["local"]))
        self.assertFalse(torch.equal(before["occupancy_parts"]["lattice"], after["occupancy_parts"]["lattice"]))

    def test_the_stencil_reads_coarse_boxes_not_samples(self):
        """A one-pixel shift of the whole image moves a box average far less than it moves a bare sample."""
        with_stencil, _ = pair()
        fill_stencil(with_stencil)
        generator = torch.Generator().manual_seed(3)
        pixels = torch.rand(1, 3, 96, 96, generator=generator)
        valid = torch.ones(1, 1, 96, 96, dtype=torch.bool)
        moved = torch.roll(pixels, shifts=2, dims=-1)
        with torch.no_grad():
            a = with_stencil(pixels, valid)["final"]
            b = with_stencil(moved, valid)["final"]
        shift = (a["occupancy_parts"]["local"]-b["occupancy_parts"]["local"]).abs().mean()
        level = a["occupancy_parts"]["local"].abs().mean()
        self.assertLess(float(shift), float(level))

    def test_the_stencil_head_leaves_the_zero_fixed_point(self):
        """The declared head must stop being identity inside the training path, not just in principle.

        A zero output projection is the V-143 fixed point *for that projection only*: its own gradient is the hidden
        activation, which is non-zero, so the projection moves on the first step and the hidden layer then carries
        gradient. Both are asserted here on the real forward, with a real optimizer, because a head that stayed zero
        would leave every stencil arm identical to its envelope arm and the block would measure nothing.
        """
        with_stencil, _ = pair()
        generator = torch.Generator().manual_seed(4)
        pixels = torch.rand(1, 3, 96, 96, generator=generator)
        valid = torch.ones(1, 1, 96, 96, dtype=torch.bool)
        optimizer = torch.optim.SGD(with_stencil.occupancy_free_stencil.parameters(), lr=.05)
        grads = []
        for _ in range(2):
            optimizer.zero_grad()
            with_stencil(pixels, valid)["final"]["occupancy_parts"]["local"].sum().backward()
            grads.append(float(with_stencil.occupancy_free_stencil[0].weight.grad.abs().sum()))
            optimizer.step()
        self.assertGreater(float(with_stencil.occupancy_free_stencil[2].weight.abs().sum()), 0.0)
        self.assertGreater(grads[1], 0.0)
        self.assertFalse(torch.equal(forward(with_stencil)["occupancy"], forward(pair()[1])["occupancy"]))

    def test_the_de_meaned_mode_carries_no_board_level(self):
        """Mode 2 is the declared constant-mode bar: what it contributes is relative, so it cannot move the level."""
        modes = []
        for mode in (1, 2):
            base = {"occupancy_lattice_free": 2, "occupancy_stack_monotone": 1}
            torch.manual_seed(7)
            model = TetrisStateNet(config(**base, occupancy_free_stencil=mode))
            fill_stencil(model)
            modes.append(forward(model)["occupancy_parts"]["local"])
        # the bar is declared over the visible board, the population every measurement scores
        visible = ModelConfig.pilot().visible_rows
        self.assertGreater(abs(float(modes[0][:, :visible].mean())), 1e-4)
        self.assertLess(abs(float(modes[1][:, :visible].mean())), 1e-6)
        self.assertGreater(float((modes[1]-modes[1].mean()).abs().max()), 0.0)

    def test_the_de_meaned_mode_keeps_its_level_at_zero_under_a_brightness_change(self):
        """The declared bar is about the level, not about insensitivity to light.

        A board-wide brightness change still moves the coarse features the stencil reads - the head is non-linear - but
        the local read it produces carries no level either way, while the pooled channel's level does move. That is the
        property the mode declares, and it is the one the curator's split scores.
        """
        base = {"occupancy_lattice_free": 2, "occupancy_stack_monotone": 1}
        torch.manual_seed(7)
        model = TetrisStateNet(config(**base, occupancy_free_stencil=2))
        fill_stencil(model)
        generator = torch.Generator().manual_seed(13)
        pixels = torch.rand(1, 3, 96, 96, generator=generator)*.5
        valid = torch.ones(1, 1, 96, 96, dtype=torch.bool)
        with torch.no_grad():
            a = model(pixels, valid)["final"]
            b = model(pixels+.4, valid)["final"]
        visible = ModelConfig.pilot().visible_rows
        for out in (a, b):
            self.assertLess(abs(float(out["occupancy_parts"]["local"][:, :visible].mean())), 1e-6)
        # the stencil does see the light - what it cannot do is turn that into a level
        self.assertGreater(float((a["occupancy_parts"]["local"][:, :visible]
                                  - b["occupancy_parts"]["local"][:, :visible]).abs().max()), 1e-5)

    def test_the_stencil_is_deterministic(self):
        with_stencil, _ = pair()
        fill_stencil(with_stencil)
        self.assertTrue(torch.equal(forward(with_stencil)["occupancy"], forward(with_stencil)["occupancy"]))

    def test_arms_without_the_stencil_carry_no_stencil_parameters(self):
        _, without = pair()
        self.assertIsNone(without.occupancy_free_stencil)
        names = [name for name, _ in without.named_parameters()]
        self.assertFalse([name for name in names if "free_stencil" in name])

    def test_the_read_without_the_stencil_is_identical_across_the_two_arms(self):
        """Constructing the head last is what keeps an arm's draw independent of whether it carries the stencil."""
        torch.manual_seed(19)
        a = TetrisStateNet(config(occupancy_lattice_free=2, occupancy_stack_monotone=1))
        torch.manual_seed(19)
        b = TetrisStateNet(config(occupancy_lattice_free=2, occupancy_stack_monotone=1))
        for (name, left), (other, right) in zip(a.named_parameters(), b.named_parameters()):
            self.assertEqual(name, other)
            self.assertTrue(torch.equal(left, right))


class ContrastReadTest(unittest.TestCase):
    """M23: the scale-invariant contrast read is invariant by construction, and proved so here.

    The property is a statement about the *coarse map* the read consumes, not about pixels: an encoder is non-linear, so
    a brightness change in the capture is not an affine change in the map, and claiming otherwise would be false. What the
    read declares - and what these tests hold it to - is that any affine change *of the coarse values* leaves it unchanged
    while a genuine local change moves it.
    """

    def coarse(self, seed=3, shape=(1, 4, 26, 12)):
        generator = torch.Generator().manual_seed(seed)
        return torch.rand(*shape, generator=generator)

    def test_an_offset_of_every_coarse_value_leaves_the_contrast_unchanged(self):
        grid = self.coarse()
        base = stencil_contrast(grid)
        for offset in (.4, -3.0, 12.5):
            self.assertLess(float((base-stencil_contrast(grid+offset)).abs().max()), 1e-4, offset)

    def test_rescaling_every_coarse_value_leaves_the_contrast_unchanged(self):
        grid = self.coarse()
        base = stencil_contrast(grid)
        for factor in (2.5, .01, 40.0):
            self.assertLess(float((base-stencil_contrast(grid*factor)).abs().max()), 1e-3, factor)

    def test_a_genuine_local_change_does_move_it(self):
        """The invariance must not be vacuity: a real local difference has to survive it."""
        grid = self.coarse()
        base = stencil_contrast(grid)
        moved = grid.clone()
        moved[:, :, 3:6, 2:5] += .5
        self.assertGreater(float((base-stencil_contrast(moved)).abs().max()), 1e-2)
        self.assertGreater(float((base-stencil_contrast(grid.roll(2, dims=-1))).abs().max()), 1e-2)

    def test_a_uniform_board_has_exactly_no_contrast(self):
        grid = self.coarse()
        for value in (0.0, 3.0, -7.5):
            self.assertEqual(float(stencil_contrast(torch.full_like(grid, value)).abs().max()), 0.0, value)

    def test_the_declared_limit_is_inactive_on_this_population(self):
        """The clamp is a numerical guard, not the mechanism; if it bit, the read would stop being invariant."""
        largest = float(stencil_contrast(self.coarse(seed=5, shape=(1, 8, 26, 12))).abs().max())
        self.assertLess(largest, CONTRAST_LIMIT/4)

    def test_the_read_through_the_head_is_invariant_too(self):
        base = {"occupancy_lattice_free": 2, "occupancy_stack_monotone": 1}
        torch.manual_seed(7)
        model = TetrisStateNet(config(**base, occupancy_free_stencil=3))
        fill_stencil(model)
        dimension = ModelConfig.pilot().dimension
        grid = self.coarse(shape=(1, dimension, 26, 12))
        rows, columns = grid.shape[-2]-2, grid.shape[-1]-2

        def read(values):
            taps = (values[:, :, 1:-1, 1:-1], values[:, :, :-2, 1:-1], values[:, :, 2:, 1:-1],
                    values[:, :, 1:-1, :-2], values[:, :, 1:-1, 2:])
            stencil = torch.cat([tap.permute(0, 2, 3, 1).reshape(1, rows*columns, -1) for tap in taps], -1)
            row_ids = torch.arange(rows)
            column_ids = torch.arange(columns)
            where = (model.row_embedding(row_ids)[:, None, :]
                     + model.column_embedding(column_ids)[None, :, :]).reshape(1, rows*columns, -1)
            with torch.no_grad():
                return model.occupancy_free_stencil(torch.cat((stencil, where), -1)).reshape(1, rows, columns)

        base_read = read(stencil_contrast(grid))
        for changed in (grid+.7, grid*3.0, grid*3.0+.7):
            self.assertLess(float((base_read-read(stencil_contrast(changed))).abs().max()), 1e-6)

    def test_the_declared_mode_carries_no_level_in_the_scored_window(self):
        base = {"occupancy_lattice_free": 2, "occupancy_stack_monotone": 1}
        torch.manual_seed(7)
        model = TetrisStateNet(config(**base, occupancy_free_stencil=3))
        fill_stencil(model)
        generator = torch.Generator().manual_seed(11)
        pixels = torch.rand(1, 3, 96, 96, generator=generator)
        valid = torch.ones(1, 1, 96, 96, dtype=torch.bool)
        with torch.no_grad():
            local = model(pixels, valid)["final"]["occupancy_parts"]["local"]
        visible = ModelConfig.pilot().visible_rows
        window = local[:, :visible]
        self.assertLess(abs(float(window.mean()))/max(1e-9, float(window.abs().mean())), 1e-6)

    def test_the_mode_is_a_declared_value(self):
        for value in (0, 1, 2, 3):
            config(occupancy_lattice_free=2, occupancy_free_stencil=value)
        for value in (-1, 4, 3.0, True):
            with self.assertRaises((ValueError, TypeError)):
                config(occupancy_lattice_free=2, occupancy_free_stencil=value)


@unittest.skipUnless(SCOPE.exists(), "the M22 curation has not run yet")
class RecordTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scope = json.loads(SCOPE.read_text(encoding="utf-8"))
        cls.doc = DOC.read_text(encoding="utf-8")

    def test_every_declared_arm_is_recorded(self):
        self.assertEqual(sorted(self.scope["arms"]), sorted(ARMS))
        for name, (free, monotone, crossing, stencil) in ARMS.items():
            self.assertEqual(self.scope["arms"][name]["declared"],
                             {"occupancy_lattice_free": free, "occupancy_stack_monotone": monotone,
                              "occupancy_crossing_threshold": crossing, "occupancy_free_stencil": stencil})

    def test_the_control_reproduces_and_the_unchanged_arms_do(self):
        control = self.scope["arms"]["control"]
        self.assertEqual(control["auc"], 0.8226433610683003)
        self.assertEqual(control["f1_at_zero"], 0.381640)
        self.assertTrue(self.scope["reproduction"]["envelope_reproduces_m20"])
        self.assertTrue(self.scope["reproduction"]["crossing_reproduces_m21"])

    def test_every_free_channel_arm_carries_the_stencil_split(self):
        for name, (free, _, _, stencil) in ARMS.items():
            if free != 2:
                continue
            channels = self.scope["channels"][name]
            self.assertIn("summed", channels)
            self.assertEqual("local_only" in channels, bool(stencil))
            self.assertEqual(("free_without_local" in channels), bool(stencil))

    def test_every_arm_is_decomposed_and_its_chain_rederives(self):
        for name in ARMS:
            block = self.scope["decomposition"][name]
            self.assertTrue(block["final_stage_matches_the_arm"])
            self.assertEqual(block["stages"][0]["stage"], "raw_sum")
            self.assertEqual(block["raising_stage"], "monotone" if ARMS[name][1] else None)

    def test_the_ablation_is_recorded_with_its_verdict(self):
        ablation = self.scope["ablation"]
        for key in ("in_loop", "post_hoc", "trained_without_rule", "in_loop_minus_post_hoc_f1", "verdict"):
            self.assertIn(key, ablation)
        self.assertIn(ablation["verdict"], ("in-loop-training-is-what-makes-the-rule-work",
                                            "post-hoc-application-matches-the-rule"))

    def test_the_doc_carries_the_measured_numbers(self):
        """Each arm's numbers appear in the document that owns that arm, never moved to fit a narrative."""
        contrast = DOC_CONTRAST.read_text(encoding="utf-8")
        for name, arm in self.scope["arms"].items():
            text = self.doc if name in EARLIER else contrast
            self.assertIn(f"{arm['auc']:.6f}", text, name)
            self.assertIn(f"{arm['f1_at_zero']:.6f}", text, name)
        for name, reading in self.scope["readings"].items():
            text = self.doc if name in EARLIER else contrast
            self.assertIn(reading["reading"], text, name)


if __name__ == "__main__":
    unittest.main()
