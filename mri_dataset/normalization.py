# masked z-score standardization.

import torch
from torch import Tensor


def masked_standardize(images: Tensor, masks: Tensor, eps: float = 1e-6) -> Tensor:

    images = images.mul(masks)
    N = masks.sum(dim=(2, 3, 4), keepdim=True).clamp_min(1.0)
    means = images.sum(dim=(2, 3, 4), keepdim=True) / N
    var = images.pow(2).sum(dim=(2, 3, 4), keepdim=True) / N - means.pow(2)
    stds = torch.sqrt(var + eps)
    images = images.sub(means)
    images = images.div(stds)
    images = images.mul(masks)
    return images
