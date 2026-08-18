"""
Message Aggregator Module

Reference:
    - https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/tgn.html

NOTE (local patch): upstream imports ``torch_scatter.scatter_max``, which has no
prebuilt wheels for newer torch releases on this environment.  ``LastAggregator``
below re-implements the same "latest timestamp per node, smallest row wins ties"
semantics with torch-native ``scatter_reduce``.  Behavior is equivalent for
distinct timestamps; timestamp ties pick the lowest message row.
"""


import torch
from torch import Tensor
from torch_geometric.utils import scatter


def _scatter_argmax(values: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """Argmax of ``values`` per index; ties resolve to the smallest row index.

    Computed in double: timestamps arrive as int64 (``last_update``) or floats,
    and float32 would lose precision on large integer timestamps.
    """
    values = values.double()
    sentinel = values.shape[0]
    init = torch.full((dim_size,), -float("inf"), device=values.device,
                      dtype=torch.double)
    tmax = torch.scatter_reduce(init, 0, index, values, reduce="amax",
                                include_self=True)
    tie = values == tmax[index]
    rows = torch.where(tie, torch.arange(sentinel, device=values.device),
                       torch.full_like(index, sentinel))
    init_rows = torch.full((dim_size,), sentinel, dtype=torch.long,
                           device=values.device)
    best = torch.scatter_reduce(init_rows, 0, index, rows, reduce="amin",
                                include_self=True)
    return best.long()


class LastAggregator(torch.nn.Module):
    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int):
        argmax = _scatter_argmax(t, index, dim_size)
        out = msg.new_zeros((dim_size, msg.size(-1)))
        mask = argmax < msg.size(0)  # Filter items with at least one entry.
        out[mask] = msg[argmax[mask]]
        return out


class MeanAggregator(torch.nn.Module):
    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int):
        return scatter(msg, index, dim=0, dim_size=dim_size, reduce="mean")
