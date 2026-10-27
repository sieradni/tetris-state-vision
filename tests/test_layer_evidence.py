"""Optional local role evidence: real-raster uncertainty and legacy math gates."""
from dataclasses import replace
import importlib
import importlib.util
import inspect
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from tetris_vision.layout import Layout
from tetris_vision.render import Appearance, Pose, render, sample_scene
from tetris_vision.state import Cell

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from tetris_vision.checkpoint import load_checkpoint, save_checkpoint
    from tetris_vision.layer_evidence import dense_targets, grid_metrics
    from tetris_vision.losses import joint_loss, targets_for
    from tetris_vision.model import ModelConfig, TetrisStateNet
    from tetris_vision.preprocess import prepare_image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from source_snapshot import catalogue_digest, sha


def frame(active=Pose("T", 3, 14), ghost=None, style=Appearance()):
    return render(replace(sample_scene(4, "empty"), active=active, ghost=ghost), style)


def prepare(result):
    prepared = prepare_image(result.image, max_side=384)
    target = targets_for(result.state, prepared, layer_evidence=True)
    return prepared, target


@unittest.skipUnless(HAS_TORCH, "install the optional train dependency for layer-evidence gates")
class LayerEvidenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(44)

    def model(self, **kwargs):
        return TetrisStateNet(replace(ModelConfig.pilot(), layer_evidence=True, **kwargs))

    def test_default_off_and_shared_seed_preserve_common_parameters(self):
        torch.manual_seed(44)
        plain = TetrisStateNet(ModelConfig.pilot())
        torch.manual_seed(44)
        dense = self.model()
        self.assertIsNone(plain.layer_evidence)
        self.assertTrue(all(torch.equal(value, dense.state_dict()[key]) for key, value in plain.state_dict().items()))
        with self.assertRaises(ValueError):
            replace(ModelConfig.pilot(), layer_evidence=1)

    def test_default_off_matches_verified_original_archive_tensors_targets_losses(self):
        import json
        code = ROOT/"tests/fixtures/m2_original"
        archive = json.loads((code/"source-manifest.json").read_text())
        self.assertEqual(catalogue_digest(archive["files"]), archive["catalogue_sha256"])
        self.assertEqual(archive["catalogue_sha256"], "121fd05a2f75e30b0c738dac592c67c6d015aa6043a43e4032d91ac662de794b")
        for name in ("__init__.py", "model.py", "losses.py", "state.py", "shapes.py"):
            self.assertEqual(sha(code/name), archive["files"]["tetris_vision/"+name])
        alias = "legacy_layer_evidence_reference"
        with patch.object(sys, "dont_write_bytecode", True):
            spec = importlib.util.spec_from_file_location(alias, code/"__init__.py", submodule_search_locations=[str(code)])
            package = importlib.util.module_from_spec(spec)
            sys.modules[alias] = package
            spec.loader.exec_module(package)
            old_model = importlib.import_module(alias+".model")
            old_losses = importlib.import_module(alias+".losses")
        self.addCleanup(lambda: [sys.modules.pop(key, None) for key in list(sys.modules) if key.startswith(alias)])
        torch.manual_seed(72)
        before = old_model.TetrisStateNet(old_model.ModelConfig.pilot()).eval()
        torch.manual_seed(72)
        after = TetrisStateNet(ModelConfig.pilot()).eval()
        self.assertEqual(before.state_dict().keys(), after.state_dict().keys())
        self.assertTrue(all(torch.equal(value, after.state_dict()[key]) for key, value in before.state_dict().items()))
        result = frame()
        prepared = prepare_image(result.image, max_side=384)
        old_target, target = old_losses.targets_for(result.state, prepared), targets_for(result.state, prepared)
        self.assertEqual(old_target.keys(), target.keys())
        self.assertTrue(all(torch.equal(value, target[key]) for key, value in old_target.items()))
        with torch.no_grad():
            left, right = before(prepared.pixels[None], prepared.valid[None]), after(prepared.pixels[None], prepared.valid[None])
            for first, second in zip(left["stages"], right["stages"]):
                self.assertEqual(first.keys(), second.keys())
                self.assertTrue(all(torch.equal(value, second[key]) for key, value in first.items()))
            old_loss, old_terms = old_losses.joint_loss(left, old_target)
            loss, terms = joint_loss(right, target)
        self.assertTrue(torch.equal(old_loss, loss))
        self.assertEqual(old_terms.keys(), terms.keys())
        self.assertTrue(all(torch.equal(value, terms[key]) for key, value in old_terms.items()))

    def test_forward_is_pixel_only_with_finite_per_stage_visible_maps(self):
        model = self.model().eval()
        self.assertEqual(list(inspect.signature(model.forward).parameters), ["pixels", "valid"])
        pixels, valid = torch.rand(1, 3, 96, 112), torch.ones(1, 1, 96, 112, dtype=torch.bool)
        with torch.no_grad():
            output = model(pixels, valid)
        for stage in output["stages"]:
            self.assertEqual(stage["layer_cells"].shape, (1, 2, 20, 10))
            self.assertTrue(all(bool(torch.isfinite(value).all()) for value in stage.values()))

    def test_identical_invisible_role_rasters_have_identical_unknown_targets(self):
        for role in ("active", "ghost"):
            pose = Pose("T", 3, 14) if role == "active" else Pose("I", 3, 10)
            scene = replace(sample_scene(4, "empty"), active=None, ghost=None)
            style = Appearance(**{role+"_alpha": 0})
            a = render(replace(scene, **{role: pose}), style)
            b = render(scene, style)
            self.assertEqual(a.image.tobytes(), b.image.tobytes())
            _, left = prepare(a)
            _, right = prepare(b)
            for key in ("occupied", "known"):
                self.assertTrue(torch.equal(left["layer_cells"][key], right["layer_cells"][key]))
            self.assertFalse(bool(left["layer_cells"]["known"][0, ("active", "ghost").index(role)].any()))

    def test_identical_unregistered_crops_cannot_recover_hidden_pose(self):
        style = Appearance(crop=(90, 250, 480, 540))
        a, b = frame(Pose("T", 3, 14), style=style), frame(Pose("I", 3, 18), style=style)
        self.assertEqual(a.image.tobytes(), b.image.tobytes())
        _, left = prepare(a)
        _, right = prepare(b)
        self.assertFalse(bool(left["registered"].any()))
        self.assertFalse(bool(left["layer_cells"]["known"].any()))
        self.assertTrue(torch.equal(left["layer_cells"]["occupied"], right["layer_cells"]["occupied"]))

    def test_known_absence_and_observed_visible_denominators_are_separate(self):
        _, target = prepare(frame(None))
        label = target["layer_cells"]
        self.assertTrue(bool(label["known"].all()))
        self.assertEqual(int(label["occupied"].sum()), 0)
        metric = grid_metrics(torch.full_like(label["occupied"], -1), label, target["layer_type"])
        self.assertEqual(metric["dense_layer_grids_exact"], 2)
        self.assertEqual(metric["dense_observed_visible_grids_known"], 0)

    def test_identified_model_outputs_are_not_new_observation_labels(self):
        result = frame()
        prepared = prepare_image(result.image, max_side=384)
        with self.assertRaisesRegex(ValueError, "model output"):
            targets_for(replace(result.state, model_id="unreviewed-prediction"), prepared, layer_evidence=True)

    def test_supported_projection_allows_overlap_without_modifying_locked_targets(self):
        # Logical public-state projection fixture; renderer controls observability.
        result = frame()
        state = replace(result.state, ghost=result.state.active)
        prepared = prepare_image(result.image, max_side=384)
        base = targets_for(state, prepared)
        before = base["occupied"].clone()
        dense = dense_targets(state, base, torch.ones_like(base["known"]))
        self.assertTrue(torch.equal(dense["occupied"][:, 0], dense["occupied"][:, 1]))
        self.assertEqual(int(dense["occupied"].sum()), 8)
        self.assertTrue(torch.equal(base["occupied"], before))

    def test_visible_role_supervision_is_independent_of_locked_certainty(self):
        result = frame()
        board = replace(result.state.board, rows=tuple(tuple(Cell(None) for _ in range(10)) for _ in range(20)))
        state = replace(result.state, board=board, decision="needs_review")
        prepared = prepare_image(result.image, max_side=384)
        target = targets_for(state, prepared, layer_evidence=True)
        self.assertFalse(bool(target["known"].any()))
        self.assertTrue(bool(target["layer_cells"]["known"][0, 0].all()))
        self.assertEqual(int(target["layer_cells"]["occupied"][0, 0].sum()), 4)

    def test_invalid_prepared_cell_is_not_supervised_as_role_absence(self):
        result = frame()
        prepared = prepare_image(result.image, max_side=384)
        x, y = result.state.board.lattice.transformed(prepared.scale_x, prepared.scale_y).center(0, 0)
        valid = prepared.valid.clone()
        valid[:, round(y), round(x)] = False
        target = targets_for(result.state, replace(prepared, valid=valid), layer_evidence=True)
        self.assertFalse(bool(target["layer_cells"]["known"][0, 0, 0, 0]))
        self.assertTrue(bool(target["layer_known"][0, 0]))

    def test_visible_grid_does_not_relabel_a_supported_spawn_piece_absent(self):
        result = frame(Pose("I", 3, 20))
        self.assertEqual(result.state.active.status, "observed")
        _, target = prepare(result)
        self.assertGreater(int(target["layer_type"][0, 0]), 0)
        self.assertTrue(bool(target["pose_known"][0, 0]))
        self.assertEqual(int(target["layer_cells"]["occupied"][0, 0].sum()), 0)

    def test_dense_loss_alone_reaches_encoder_sampling_and_geometry(self):
        prepared, target = prepare(frame())
        model = self.model()
        _, terms = joint_loss(model(prepared.pixels[None], prepared.valid[None]), target)
        terms["layer_cells"].backward()
        for parameter in (model.layer_evidence.cells.weight, model.encoder.stages[0][0].weight, model.initial_box[1].weight):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()))

    def test_unknown_roles_have_zero_auxiliary_loss_and_gradient(self):
        prepared, target = prepare(frame(Pose("T", 3, 14), Pose("I", 3, 10), Appearance(active_alpha=0, ghost_alpha=0)))
        self.assertFalse(bool(target["layer_cells"]["known"].any()))
        model = self.model()
        _, terms = joint_loss(model(prepared.pixels[None], prepared.valid[None]), target)
        self.assertEqual(float(terms["layer_cells"].detach()), 0)
        terms["layer_cells"].backward()
        self.assertEqual(float(model.layer_evidence.cells.weight.grad.abs().sum()), 0)

    def test_enabled_checkpoint_restores_all_outputs_and_configuration(self):
        model = self.model().eval()
        pixels, valid = torch.rand(1, 3, 96, 112), torch.ones(1, 1, 96, 112, dtype=torch.bool)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"model"
            record = save_checkpoint(path, model, {"test": True})
            self.assertIs(record["config"]["layer_evidence"], True)
            restored, _ = load_checkpoint(path)
            with torch.no_grad():
                left, right = model(pixels, valid)["final"], restored.eval()(pixels, valid)["final"]
            self.assertTrue(all(torch.equal(value, right[key]) for key, value in left.items()))

    def test_component_and_dense_paths_jointly_backpropagate(self):
        layout = Layout(canvas=(208, 256), board_left=8, board_bottom=248, cell_size=10,
                        queue_origin=(122, 8), hold_origin=(122, 184), icon_size=6)
        scene = replace(sample_scene(4), queue=("T", "I", "O"), hold="L")
        result = render(scene, layout=layout)
        prepared = prepare_image(result.image, max_side=256)
        target = targets_for(result.state, prepared, component_capacity=4, layer_evidence=True)
        model = self.model(component_capacity=4)
        loss, terms = joint_loss(model(prepared.pixels[None], prepared.valid[None]), target)
        self.assertIn("components", terms)
        self.assertIn("layer_cells", terms)
        loss.backward()
        self.assertGreater(float(model.layer_evidence.cells.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.components.queue_tokens.weight.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
