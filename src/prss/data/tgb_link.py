"""TGB link-prediction dataset wrapper (single tgbl-* dataset at a time).

The wrapper is name-parameterized: tgbl-wiki / tgbl-uci / tgbl-enron share one
code path (same CSV layout, 172-dim edge messages, no node features, mrr metric).
"""

import os
from typing import Dict, Optional, Tuple

import torch


def _set_tgb_proj_dir(parent: str) -> None:
    """Point TGB's PROJ_DIR at ``parent`` (must end with the path separator).

    TGB builds its data root as a plain string concatenation ``PROJ_DIR + root``
    where PROJ_DIR defaults to the installed package dir, so an absolute ``root``
    would be mangled.  By setting PROJ_DIR to the parent of our absolute root and
    passing only the directory name, the concatenation lands exactly on our
    directory on any machine, regardless of cwd.
    """
    import tgb.linkproppred.dataset as _dataset

    parent = parent.rstrip("/\\")
    if not parent:
        parent = "."
    _dataset.PROJ_DIR = parent + "/"


class TGBLinkDataset:
    """Thin, leak-safe wrapper around ``PyGLinkPropPredDataset``."""

    def __init__(self, name: str = "tgbl-wiki", root: Optional[str] = None,
                 device: Optional[torch.device] = None):
        from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset

        root = root or os.environ.get("TGB_ROOT", "datasets")
        abs_root = os.path.abspath(root)
        _set_tgb_proj_dir(os.path.dirname(abs_root))
        root = os.path.basename(abs_root)

        self.name = name
        self._ds = PyGLinkPropPredDataset(name=name, root=root)
        self._data = self._ds.get_TemporalData()
        if device is not None:
            self._data = self._data.to(device)
        self.device = device
        self.num_nodes = int(self._data.num_nodes)
        self.msg_dim = int(self._data.msg.size(-1))
        self.eval_metric = self._ds.eval_metric
        self.train_mask = self._ds.train_mask
        self.val_mask = self._ds.val_mask
        self.test_mask = self._ds.test_mask
        self.min_dst_idx = int(self._data.dst.min())
        self.max_dst_idx = int(self._data.dst.max())
        self._train_data = self._data[self.train_mask]
        self._val_data = self._data[self.val_mask]
        self._test_data = self._data[self.test_mask]

    @property
    def train_data(self):
        return self._train_data

    @property
    def val_data(self):
        return self._val_data

    @property
    def test_data(self):
        return self._test_data

    def load_val_ns(self):
        self._ds.load_val_ns()

    def load_test_ns(self):
        self._ds.load_test_ns()

    def query_batch(self, pos_src, pos_dst, pos_t, split_mode):
        return self._ds.negative_sampler.query_batch(pos_src, pos_dst, pos_t,
                                                     split_mode=split_mode)

    @property
    def evaluator(self):
        from tgb.linkproppred.evaluate import Evaluator
        return Evaluator(name=self.name)

    def build_loader(self, split: str, batch_size: int):
        from torch_geometric.loader import TemporalDataLoader
        data = {"train": self._train_data, "val": self._val_data,
                "test": self._test_data}[split]
        return TemporalDataLoader(data, batch_size=batch_size)

    def time_stats(self) -> Tuple[float, float]:
        """(mean, std) of log1p timestamps over the training stream (root metadata norm)."""
        t = self._train_data.t.double().cpu()
        logt = torch.log1p(t.clamp_min(0))
        mean = float(logt.mean())
        std = float(logt.std(unbiased=False)) + 1e-8
        return mean, std

    def sanity_check(self) -> Dict:
        checks = {
            "name": self.name,
            "num_nodes": self.num_nodes,
            "num_edges": int(self._data.src.numel()),
            "msg_dim": self.msg_dim,
            "node_feat_is_none": self._ds.node_feat is None,
            "eval_metric": self.eval_metric,
            "train/val/test masks disjoint": bool(
                (self.train_mask & self.val_mask).sum() == 0 and
                (self.train_mask & self.test_mask).sum() == 0 and
                (self.val_mask & self.test_mask).sum() == 0),
            "timestamps_nonnegative": bool((self._data.t >= 0).all()),
        }
        if self.name == "tgbl-wiki":
            checks.update({
                "wiki_num_nodes_expected_9227": self.num_nodes == 9227,
                "wiki_num_edges_expected_157474": int(self._data.src.numel()) == 157474,
                "wiki_msg_dim_expected_172": self.msg_dim == 172,
            })
        return checks
