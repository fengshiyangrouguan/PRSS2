"""TGB tgbl-wiki data-layer gates (plan step 1).

Verifies the data-layer contract on real tgbl-wiki files:
* 157,474 events / 9,227 TGB nodes / 172-dim msg;
* official train/val/test masks pairwise disjoint, chronological;
* internal (+1) node/edge ids in range (TGN padding sentinel 0);
* edge-feature row 0 is zero padding;
* val/test negative samplers load and return in-range dst.

Runs only on a machine that has the data (server).  Requires py-tgb + pyg.
"""

import os
import sys
import unittest
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rpbe.data.tgb_link import TGBLinkDataset


def _skip_if_no_data():
    try:
        ds = TGBLinkDataset(root="datasets")
        ds.sanity_check()
        return False
    except Exception as e:  # pragma: no cover - environment dependent
        return True


class TestTGBLinkData(unittest.TestCase):
    def test_wiki_scale_and_masks(self):
        ds = TGBLinkDataset(root="datasets")
        c = ds.sanity_check()
        self.assertEqual(c["name"], "tgbl-wiki")
        self.assertEqual(c["n_tgb_nodes"], 9227)
        self.assertEqual(c["n_events"], 157474)
        self.assertEqual(c["msg_dim"], 172)
        self.assertTrue(c["masks_disjoint"])
        self.assertTrue(c["timestamps_nondecreasing"])
        self.assertTrue(c["edge_feat_row0_zero"])
        self.assertTrue(c["internal_ids_gt0"])
        self.assertTrue(c["internal_id_in_range"])
        # internal node table has 9227 + padding row
        self.assertEqual(ds.n_internal_nodes, 9228)
        # every event in train/val/test adds up to full
        self.assertEqual(len(ds.train.sources) + len(ds.val.sources)
                         + len(ds.test.sources), 157474)
        # labels are all positive (the stream is the positive edge set)
        self.assertTrue((ds.train.labels == 1.0).all())

    def test_negative_sampler_loads_and_in_range(self):
        ds = TGBLinkDataset(root="datasets")
        ds.load_val_ns()
        # negative sampler indexes by RAW (0-based) ids
        raw_src, raw_dst, raw_t = ds.raw_split("val")
        idxs = list(range(0, len(raw_src), 2000))
        pos_src = [int(raw_src[i]) for i in idxs]
        pos_dst = [int(raw_dst[i]) for i in idxs]
        pos_t = [float(raw_t[i]) for i in idxs]
        neg = ds.query_negatives(np.asarray(pos_src, dtype=np.int64),
                                 np.asarray(pos_dst, dtype=np.int64),
                                 np.asarray(pos_t, dtype=np.float64),
                                 split_mode="val")
        self.assertIsInstance(neg, list)
        self.assertEqual(len(neg), len(pos_src))
        flat = [x for sub in neg for x in sub]
        arr = np.asarray(flat, dtype=np.int64)  # raw ids
        self.assertTrue((arr >= 0).all())
        self.assertLess(arr.max(), ds.n_nodes_tgb)
        # internal (+1) mapping keeps negatives in range too
        self.assertLess((arr + 1).max(), ds.n_internal_nodes)


if __name__ == "__main__":
    unittest.main()
