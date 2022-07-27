import torch
import numpy as np


def ar2ten(array, device, dtype=None):
    if isinstance(array, list) or isinstance(array, dict):
        return array

    if isinstance(array, np.ndarray):
        tensor = torch.from_numpy(array).to(device)
    else:
        tensor = torch.tensor(array).to(device)
    if dtype is not None:
        tensor = tensor.to(dtype)
    return tensor


def ten2ar(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    elif torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy()
    elif np.isscalar(tensor):
        return tensor
    elif hasattr(tensor, "to_numpy"):
        return tensor.to_numpy()
    else:
        import pdb

        pdb.set_trace()
        raise ValueError("input to ten2ar cannot be converted to numpy array")
