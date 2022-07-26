import torch
from typing import Sized, List


def padded_tensor(
    items,
    pad_idx,
    left_padded,
    max_len=None,
):
    # number of items
    n = len(items)
    # length of each item
    lens: List[int] = [len(item) for item in items]  # type: ignore
    # max in time dimension
    t = max(lens) if max_len is None else max_len

    # if input tensors are empty, we should expand to nulls
    t = max(t, 1)

    if isinstance(items[0], torch.Tensor):
        # keep type of input tensors, they may already be cuda ones
        output = items[0].new(n, t)  # type: ignore
    else:
        output = torch.LongTensor(n, t)  # type: ignore
    output.fill_(pad_idx)

    for i, (item, length) in enumerate(zip(items, lens)):
        if length == 0:
            # skip empty items
            continue
        if not isinstance(item, torch.Tensor):
            # put non-tensors into a tensor
            item = torch.LongTensor(item)  # type: ignore
        if left_padded:
            # place at end
            output[i, t - length :] = item
        else:
            # place at beginning
            output[i, :length] = item

    return output, lens


def padded_3d(
    tensors,
    pad_idx,
    dtype=torch.long,
):
    """
    Make 3D padded tensor for list of lists of 1D tensors or lists.
    Will keep items on the same device as originally.
    :param tensors:
        list of lists of 1D tensors (or lists)
    :param pad_idx:
        padding to fill tensor with
    :param bool fp16friendly:
        if True, pads the final dimension to be a multiple of 8.
    :returns:
        3D tensor with the maximum dimensions of the inputs
    """
    a = len(tensors)
    b = max(len(row) for row in tensors)  # type: ignore
    c = max(len(item) for row in tensors for item in row)  # type: ignore

    c = max(c, 1)

    dev = tensors[0][0].device
    output = torch.full((a, b, c), pad_idx, dtype=dtype, device=dev)

    for i, row in enumerate(tensors):
        item: Sized
        for j, item in enumerate(row):  # type: ignore
            if len(item) == 0:
                continue
            if not isinstance(item, torch.Tensor):
                item = torch.as_tensor(item, dtype=dtype)
            output[i, j, : len(item)] = item

    return output
