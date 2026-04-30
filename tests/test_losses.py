import math

import torch

from imgattck.losses import target_probability_loss


def test_target_probability_loss_uses_softmax_mass():
    logits = torch.tensor([[0.0, 2.0, 1.0]])

    loss, metrics = target_probability_loss(logits, [1, 2])

    expected_probability = float(torch.softmax(logits, dim=-1)[0, [1, 2]].sum())
    assert math.isclose(metrics.target_probability, expected_probability, rel_tol=1e-6)
    assert math.isclose(float(loss), -math.log(expected_probability), rel_tol=1e-6)
