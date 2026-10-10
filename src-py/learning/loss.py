import torch.nn.functional as F

def reconstruction_loss(prediction, target, edge_weight=1.0):
    # Measure target edge strength across RGB channels.
    dx = (target[..., 1:] - target[..., :-1]).abs().mean(
        dim=1, keepdim=True
    )
    dy = (target[..., 1:, :] - target[..., :-1, :]).abs().mean(
        dim=1, keepdim=True
    )

    # Assign each difference to both pixels touching it.
    edges = (
        F.pad(dx, (0, 1)) + F.pad(dx, (1, 0))
        + F.pad(dy, (0, 0, 0, 1)) + F.pad(dy, (0, 0, 1, 0))
    )

    # Normalize separately for each image.
    strength = edges.amax(dim=(-2, -1), keepdim=True)
    edges = edges / strength.clamp_min(1e-6)

    weights = 1.0 + edge_weight * edges
    error = (prediction - target).abs()

    return (weights * error).sum() / (
        weights.sum() * prediction.shape[1]
    )