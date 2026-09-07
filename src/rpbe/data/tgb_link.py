"""TGB link-prediction dataset wrapper for the recursive Twitter-TGN host.

Adapts TGB 2.0 ``tgbl-wiki`` into the JODIE-style numpy streams the
multi-layer recursive host (``rpbe.hosts.jodie_tgn`` + official TGN) consumes,
while keeping TGB's official train/val/test masks, negative sampler and MRR
evaluator.

Mapping (TGN padding convention):
    internal_node_id = tgb_node_id + 1        (0 = padding sentinel)
    internal_edge_id = global_event_row + 1   (0 = padding row in edge table)

The edge feature table's row 0 is all-zero padding; rows 1.. are TGB's
172-dim ``msg``.  Node features are all-zero (TGB wiki has no node features);
we never write future messages/labels into node features.  ``src``/``dst``
can be either orientation; TGB treats the stream as a directed edge list and
the evaluator is orientation-agnostic for the positive pair.

No PyG single-layer host is introduced here — this is data only.
"""

import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from rpbe.data.jodie import JodieData


def _set_tgb_proj_dir(parent: str) -> None:
    """Point TGB's PROJ_DIR at ``parent`` (must end with the path separator).

    TGB builds its data root as a plain string concatenation ``PROJ_DIR + root``
    where PROJ_DIR defaults to the installed package dir, so an absolute ``root``
    would be mangled.
    """
    import tgb.linkproppred.dataset as _dataset

    parent = parent.rstrip("/\\")
    if not parent:
        parent = "."
    _dataset.PROJ_DIR = parent + "/"


class TGBLinkDataset:
    """Leak-safe TGB link dataset exposing JODIE-style numpy streams.

    Attributes mirror what the recursive host / training loop needs:
      train / val / test : JodieData (internal +1 ids, edge_idxs 1-based,
                           ``labels`` = 1.0 everywhere — the stream is the
                           positive edge set; negatives come from TGB sampler)
      edge_features      : [n_edges + 1, msg_dim] float32 (row 0 = zero pad)
      node_features      : [n_nodes + 1, feat_dim] float32 (row 0 = zero pad)
      n_nodes            : number of TGB nodes (+1 padding)
      msg_dim            : 172 for tgbl-wiki
      train/val/test masks kept for the official evaluator path.
    """

    def __init__(self, name: str = "tgbl-wiki", root: Optional[str] = None,
                 device: Optional[torch.device] = None,
                 node_feat_dim: int = 172):
        from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset

        root = root or os.environ.get("TGB_ROOT", "datasets")
        abs_root = os.path.abspath(root)
        _set_tgb_proj_dir(os.path.dirname(abs_root))
        root = os.path.basename(abs_root)

        self.name = name
        self._ds = PyGLinkPropPredDataset(name=name, root=root)
        data = self._ds.get_TemporalData()
        if device is not None:
            data = data.to(device)
        self._data = data
        self.device = device
        self.msg_dim = int(data.msg.size(-1))
        self.eval_metric = self._ds.eval_metric

        train_mask = np.asarray(self._ds.train_mask, dtype=bool)
        val_mask = np.asarray(self._ds.val_mask, dtype=bool)
        test_mask = np.asarray(self._ds.test_mask, dtype=bool)
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask

        # ---- raw TGB tensors (cpu)
        src = data.src.detach().cpu().numpy()
        dst = data.dst.detach().cpu().numpy()
        t = data.t.detach().cpu().numpy()
        msg = data.msg.detach().cpu().numpy() if data.msg is not None else None
        n_events = int(src.shape[0])

        # +1 internal mapping (0 = padding sentinel); RAW ids are kept for the
        # official TGB sampler / evaluator, which index by raw (0-based) ids.
        self.n_nodes_tgb = int(max(int(src.max()), int(dst.max())) + 1)
        isrc = (src + 1).astype(np.int64)
        idst = (dst + 1).astype(np.int64)
        # chronological row order == event order; internal edge id = row+1
        eidx = (np.arange(n_events) + 1).astype(np.int64)

        if msg is None:
            msg = np.zeros((n_events, 172), dtype=np.float32)
        # edge feature table: row 0 zero padding, rows 1.. = message
        edge_features = np.zeros((n_events + 1, msg.shape[-1]),
                                 dtype=np.float32)
        edge_features[1:] = np.asarray(msg, dtype=np.float32)

        # node feature table: row 0 zero padding, then zeros for each node.
        # All-zero node features match official TGN-compatible usage and never
        # leak future messages/labels.
        node_features = np.zeros((self.n_nodes_tgb + 1, int(node_feat_dim)),
                                 dtype=np.float32)

        # labels=1.0: the stream rows ARE positive edges.
        labels = np.ones(n_events, dtype=np.float64)

        self.edge_features = edge_features
        self.node_features = node_features
        self.n_internal_nodes = int(self.n_nodes_tgb + 1)

        def _split(mask):
            m = np.asarray(mask, dtype=bool)
            return JodieData(
                isrc[m], idst[m], t[m].astype(np.float64),
                eidx[m], labels[m])

        # Raw (0-based) split arrays for the official sampler / evaluator.
        self.raw_src = src.astype(np.int64)
        self.raw_dst = dst.astype(np.int64)

        self.train = _split(train_mask)
        self.val = _split(val_mask)
        self.test = _split(test_mask)
        self.full = JodieData(isrc, idst, t.astype(np.float64), eidx, labels)

        # ---- negative sampler / evaluator are TGB's own
        self._ns_loaded = {"val": False, "test": False}

    # ------------------------------------------------------------- TGB APIs
    def raw_split(self, split):
        """Raw (0-based) (src, dst, t) for a split, for sampler/evaluator."""
        m = {"train": self.train_mask, "val": self.val_mask,
             "test": self.test_mask}[split]
        return (self.raw_src[m], self.raw_dst[m],
                self._data.t.detach().cpu().numpy()[m].astype(np.float64))

    def to_internal(self, raw_nodes):
        return np.asarray(raw_nodes, dtype=np.int64) + 1

    def to_raw(self, internal_nodes):
        return np.asarray(internal_nodes, dtype=np.int64) - 1

    def load_val_ns(self):
        self._ds.load_val_ns()
        self._ns_loaded["val"] = True

    def load_test_ns(self):
        self._ds.load_test_ns()
        self._ns_loaded["test"] = True

    def query_negatives(self, pos_src, pos_dst, pos_t, split_mode):
        """Official TGB negative batch.  Args/returns use RAW (0-based) ids."""
        return self._ds.negative_sampler.query_batch(
            pos_src, pos_dst, pos_t, split_mode=split_mode)

    @property
    def evaluator(self):
        from tgb.linkproppred.evaluate import Evaluator
        return Evaluator(name=self.name)

    def splits(self):
        return self.full, self.train, self.val, self.test

    def time_stats(self) -> Tuple[float, float]:
        """(mean, std) of log1p timestamps over the train stream."""
        t = torch.from_numpy(self.train.timestamps).float()
        logt = torch.log1p(t.clamp_min(0))
        return float(logt.mean()), float(logt.std(unbiased=False)) + 1e-8

    # -------------------------------------------------------------- sanity
    def sanity_check(self) -> Dict:
        def _overlap(a, b):
            return bool(np.logical_and(a, b).sum() > 0)

        checks = {
            "name": self.name,
            "n_tgb_nodes": self.n_nodes_tgb,
            "n_internal_nodes": self.n_internal_nodes,
            "n_events": int(len(self.full.sources)),
            "msg_dim": self.msg_dim,
            "eval_metric": self.eval_metric,
            "n_train": int(len(self.train.sources)),
            "n_val": int(len(self.val.sources)),
            "n_test": int(len(self.test.sources)),
            "masks_disjoint": not (_overlap(self.train_mask, self.val_mask)
                                   or _overlap(self.train_mask, self.test_mask)
                                   or _overlap(self.val_mask, self.test_mask)),
            "timestamps_nondecreasing": bool(
                (np.diff(self.full.timestamps) >= 0).all()),
            "edge_feat_row0_zero": bool((self.edge_features[0] == 0).all()),
            "internal_ids_gt0": bool(
                (self.train.sources >= 1).all() and (self.train.destinations >= 1).all()),
            "internal_id_in_range": bool(
                self.train.sources.max() < self.n_internal_nodes
                and self.train.destinations.max() < self.n_internal_nodes),
        }
        if self.name == "tgbl-wiki":
            checks.update({
                "wiki_nodes_expected_9227": self.n_nodes_tgb == 9227,
                "wiki_events_expected_157474": len(self.full.sources) == 157474,
                "wiki_msg_expected_172": self.msg_dim == 172,
            })
        return checks
