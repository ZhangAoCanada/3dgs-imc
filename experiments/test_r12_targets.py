"""CPU checks of the R1.2 target semantics and trainable derivative loss."""
import importlib.util
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cpf_targets", ROOT / "scene_experimental_branch3/cpf_targets.py")
targets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(targets)


class TargetChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_analytic_gradient_in_world_coordinates(self):
        for mode in ("normal_off", "signed_plane"):
            x = torch.randn(128, 3, dtype=torch.float64, requires_grad=True)
            p = torch.randn_like(x)
            n = torch.randn_like(x)
            value, analytic = targets.probability_targets(x, p, n, 0.7, mode)
            automatic = torch.autograd.grad(value.sum(), x)[0]
            torch.testing.assert_close(automatic, analytic, atol=1e-12, rtol=1e-12)

    def test_plane_sides_peak_tangent_and_normal_flip(self):
        x = torch.tensor([[2., 3., -0.5], [2., 3., 0.], [2., 3., 0.5]], dtype=torch.float64)
        p = torch.zeros_like(x)
        n = torch.tensor([0., 0., 1.], dtype=x.dtype).expand_as(x)
        value, gradient = targets.probability_targets(x, p, n, 0.5, "signed_plane")
        self.assertGreater(float(gradient[0, 2]), 0)
        self.assertLess(float(gradient[2, 2]), 0)
        torch.testing.assert_close(gradient[1], torch.zeros(3, dtype=x.dtype))
        self.assertEqual(float(value[1]), 1.)
        for changed_x, changed_n in ((x, -n), (x + x.new_tensor([17., -9., 0.]), n)):
            pv, pg = targets.probability_targets(changed_x, p, changed_n, 0.5, "signed_plane")
            torch.testing.assert_close(pv, value)
            torch.testing.assert_close(pg, gradient)

    def test_world_normal_for_tilted_surface(self):
        y, x = torch.meshgrid(torch.arange(5.), torch.arange(7.), indexing="ij")
        points = torch.stack((x, y, 2*x - 3*y + 4), -1)
        normals = targets.world_normals_from_points(points.reshape(-1, 3), 5, 7).reshape(5, 7, 3)
        expected = F.normalize(torch.tensor([-2., 3., 1.]), dim=0)
        torch.testing.assert_close(normals[1:-1, 1:-1], expected.expand(3, 5, 3))
        self.assertEqual(int(torch.count_nonzero(normals[0])), 0)

    def test_local_queries_have_one_nearest_support(self):
        points = torch.tensor([[-1., 0., 0., 1., 0., 0., 0., 0., 1.],
                               [1., 0., 0., 0., 1., 0., 0., 0., 1.]])
        support = targets.LocalSupport(points)
        batch = support.sample(2, 0.2, torch.tensor(-2.), torch.tensor(2.), "signed_plane")
        nearest = torch.cdist(batch["query"].detach(), points[:, :3]).argmin(-1)
        expected, grad = targets.probability_targets(batch["query"], points[nearest, :3], points[nearest, 6:], 0.2, "signed_plane")
        torch.testing.assert_close(batch["probability"], expected)
        torch.testing.assert_close(batch["gradient"], grad)
        self.assertTrue(batch["query"].requires_grad)
        self.assertFalse(batch["gradient"].requires_grad)

    def test_removal_has_no_derivative_penalty(self):
        x = torch.randn(8, 3, requires_grad=True)
        prediction = {"sigma": x[:, :1].sigmoid(), "rgb": x.sigmoid()}
        p = torch.rand(8, 1)
        rgb = torch.rand(8, 3)
        loss, terms = targets.field_losses(prediction, x, p, torch.full_like(x, float("nan")), rgb, 0.5, "normal_off")
        expected = 3000 * F.l1_loss(prediction["sigma"], p) + F.l1_loss(prediction["rgb"], rgb)
        torch.testing.assert_close(loss, expected)
        self.assertEqual(float(terms["gradient_mse_h_scaled"]), 0.)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(x.grad).all()))

    def test_derivative_loss_updates_network_parameters(self):
        x = torch.randn(32, 3, dtype=torch.float64, requires_grad=True)
        net = torch.nn.Sequential(torch.nn.Linear(3, 16), torch.nn.Tanh(), torch.nn.Linear(16, 4)).double()
        pred = net(x).sigmoid()
        p, grad = targets.probability_targets(x.detach(), torch.zeros_like(x), x.new_tensor([0., 0., 1.]), 0.5, "signed_plane")
        loss, _ = targets.field_losses({"sigma": pred[:, :1], "rgb": pred[:, 1:]}, x, p, grad, torch.zeros_like(x), 0.5, "signed_plane", probability_weight=0, color_weight=0)
        loss.backward()
        self.assertTrue(all(t.grad is not None and torch.isfinite(t.grad).all() for t in net.parameters()))
        self.assertGreater(sum(float(t.grad.norm()) for t in net.parameters()), 0)

    def test_bandwidth_rejects_zero(self):
        with self.assertRaises(ValueError):
            targets.probability_targets(torch.zeros(1, 3), torch.zeros(1, 3), torch.ones(1, 3), 0, "signed_plane")


if __name__ == "__main__":
    unittest.main(verbosity=2)
