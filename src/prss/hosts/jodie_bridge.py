"""Training-only continuation contexts for JODIE node classification
(official TGN host).

One scenario per traced interaction row: (source tree, label y in {0, 1}).
Root metadata is the normalized log-time — a single scalar — and the label
enters the losses only, never the context encoder.  The whole-tree top-down
outside pass is delegated to ``prss.auxiliary.build_auxiliary``, which skips
the d == k base interface (``tjo:layer0``) automatically.
"""

import numpy as np
import torch

from prss.auxiliary import AuxiliaryBatch, build_auxiliary
from prss.hosts.base import OutsideBridge


class JodieNodeClassificationBridge(OutsideBridge):
    """Builds the auxiliary batch from the adapter's trace plus root times/labels."""

    def __init__(self, adapter, prss_core, log_time_mean: float = 0.0,
                 log_time_std: float = 1.0):
        self.adapter = adapter
        self.prss = prss_core
        self.log_time_mean = float(log_time_mean)
        self.log_time_std = max(float(log_time_std), 1e-12)
        if prss_core.config.root_metadata_dim != 1:
            raise ValueError(
                "JODIE root_metadata_dim must be 1 (normalized log-time), got {}".format(
                    prss_core.config.root_metadata_dim))

    def build(self, root_times, root_labels) -> AuxiliaryBatch:
        """root_times / root_labels are aligned with ``trace.root_rows``.

        root_times: array-like of raw interaction timestamps (float);
        root_labels: [n_roots] float tensor of 0/1 labels.
        """
        trace = self.adapter.trace
        if trace is None or not trace.roots:
            zero = root_labels.sum() * 0.0
            return AuxiliaryBatch(zero, zero, zero, root_labels[:0],
                                  root_labels[:0], root_labels[:0], {}, {}, {})
        times = np.asarray(root_times, dtype=np.float64)
        logt = np.log1p(np.clip(times, 0.0, None))
        norm = (logt - self.log_time_mean) / self.log_time_std
        metadata = torch.as_tensor(norm, dtype=root_labels.dtype,
                                   device=root_labels.device).reshape(-1, 1)
        return build_auxiliary(self.prss, trace, metadata, root_labels,
                               response_task="binary")
