"""Consistent probability targets for the R1.2 comparison, in world coordinates.

The signed-plane target is exactly exp(-d**2/(2*h**2)), with no tangential
envelope. Queries are local to observed support. Nearest-anchor association is
fixed per query; the derivative is the derivative within that local patch.
It does not assert a globally smooth field across association boundaries.
"""
import torch
import torch.nn.functional as F


def probability_targets(query, anchor, normal, h, mode):
    if h <= 0:
        raise ValueError("CPF bandwidth h must be positive")
    offset = query - anchor
    if mode == "normal_off":
        probability = torch.exp(-offset.square().sum(-1, keepdim=True) / (2 * h * h))
        gradient = -offset * probability / (h * h)
    elif mode == "signed_plane":
        normal = F.normalize(normal, dim=-1, eps=1e-12)
        distance = (offset * normal).sum(-1, keepdim=True)
        probability = torch.exp(-distance.square() / (2 * h * h))
        gradient = -distance * probability * normal / (h * h)
    else:
        raise ValueError(f"Unknown CPF target mode: {mode}")
    return probability, gradient


def world_normals_from_points(points, height, width):
    """Perspective-correct local normals from an organized world-space cloud.

    Borders and invalid/degenerate neighborhoods have zero normals and are
    excluded by LocalSupport. The target is invariant to the orientation sign.
    """
    grid = points.reshape(height, width, 3)
    normal = torch.zeros_like(grid)
    if height < 3 or width < 3:
        return normal.reshape(-1, 3)
    dx = grid[1:-1, 2:] - grid[1:-1, :-2]
    dy = grid[2:, 1:-1] - grid[:-2, 1:-1]
    cross = torch.linalg.cross(dx, dy, dim=-1)
    valid = torch.isfinite(cross).all(-1, keepdim=True) & (cross.norm(dim=-1, keepdim=True) > 1e-12)
    cross = torch.where(valid, cross, torch.zeros_like(cross))
    normal[1:-1, 1:-1] = F.normalize(cross, dim=-1, eps=1e-12)
    return normal.reshape(-1, 3)


class LocalSupport:
    """One immutable support/association index per CPF refresh."""
    def __init__(self, points, max_support=50000):
        from scipy.spatial import cKDTree
        valid = torch.isfinite(points).all(-1) & (points[:, 6:9].norm(dim=-1) > 1e-6)
        points = points[valid].detach()
        if points.shape[0] == 0:
            raise ValueError("No finite CPF support with valid world-space normals")
        if points.shape[0] > max_support:
            points = points[torch.randperm(points.shape[0], device=points.device)[:max_support]]
        self.points = points
        self.tree = cKDTree(points[:, :3].cpu().numpy())

    def sample(self, count, h, lower, upper, mode):
        count = min(count, len(self.points))
        ids = torch.randperm(len(self.points), device=self.points.device)[:count]
        anchors = self.points[ids, :3]
        query = torch.cat([anchors, anchors + h * torch.randn_like(anchors)], dim=0)
        # Remain inside the network's coordinate bounds so clamping does not
        # force zero gradients while supervising a nonzero derivative.
        margin = (upper - lower) * 1e-6
        query = query.clamp(min=lower + margin, max=upper - margin)
        nearest = self.tree.query(query.detach().cpu().numpy(), workers=1)[1]
        matched = self.points[torch.as_tensor(nearest, device=query.device)]
        probability, gradient = probability_targets(query, matched[:, :3], matched[:, 6:9], h, mode)
        return {
            "query": query.detach().requires_grad_(True),
            "probability": probability.detach(),
            "gradient": gradient.detach(),
            "color": matched[:, 3:6].detach(),
        }


def field_losses(prediction, query, probability, gradient, color, h, mode,
                 probability_weight=3000.0, gradient_weight=100.0, color_weight=1.0):
    probability_loss = F.l1_loss(prediction["sigma"], probability)
    color_loss = F.l1_loss(prediction["rgb"], color)
    gradient_loss = probability_loss.new_zeros(())
    if mode == "signed_plane":
        predicted_gradient = torch.autograd.grad(
            prediction["sigma"].sum(), query, create_graph=True, retain_graph=True
        )[0]
        # h nondimensionalizes the vector residual. Both its sign and magnitude
        # remain supervised, including a zero derivative at d=0.
        gradient_loss = ((predicted_gradient - gradient) * h).square().sum(-1).mean()
    elif mode != "normal_off":
        raise ValueError(mode)
    total = probability_weight * probability_loss + color_weight * color_loss
    total = total + gradient_weight * gradient_loss
    return total, {"probability_l1": probability_loss.detach(),
                   "gradient_mse_h_scaled": gradient_loss.detach(),
                   "color_l1": color_loss.detach()}
