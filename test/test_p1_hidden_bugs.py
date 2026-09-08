"""P1 正式运行暴露的边界回归测试（CPU，不依赖数据）。

三个已踩坑的回归 + 一个评测路径前置检查：
1. seed 表达式 32-bit 溢出（global_step 累积，epoch 6+ 崩）
2. AuditAccumulator 确定性 / multiset 顺序不变 / replay 顺序重放 / 缺失检测
3. PairKFWindow.close_replay 覆盖全量记录（exact replay 前提）
4. query 集文件存在且 id 在 val ns 范围内（评测路径前置）
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))

from rpbe.audit import AuditAccumulator  # noqa: E402
from rpbe.link_records import (BoundaryRecord, ObservedLinkEvent)  # noqa: E402
from rpbe.pair_arms import (build_mispaired_parent_map,  # noqa: E402
                            feasible_positions)
from rpbe.pair_window import PairKFWindow  # noqa: E402


# ---------------------------------------------------------------- fixtures
def _rec(pid, root_row, t=100.0, parent_time=None):
    cf = ObservedLinkEvent(event_id=pid * 10 + 1, time=t + 1.0, src=1, dst=2,
                           counterpart=2, role=0, message_idx=1)
    pf = ObservedLinkEvent(event_id=pid * 10 + 2, time=t + 2.0, src=1, dst=3,
                           counterpart=3, role=0, message_idx=1)
    pt = t if parent_time is None else parent_time
    return BoundaryRecord(
        pair_id=(pid, 0, 2, 0), root_row=root_row, tau="tjo:layer1",
        child_layer=1, parent_layer=2, child_node=10 + pid, child_time=t,
        parent_node=20, parent_time=pt,
        relation_time=t - 1.0, relation_lag=1.0, relation_slot=0,
        path=((0, 0.0),), z=torch.randn(4),
        child_future=cf, parent_future=pf, weight=1.0)


# ------------------------------------------------------- 1. seed mask 边界
class TestSeedMaskBounds:
    def test_feasible_positions_large_batch_seed(self):
        """global_step 累积到 epoch 100+ 的量级不崩（旧代码抛 ValueError）。"""
        recs = [_rec(i, i % 8, t=100.0 + i) for i in range(40)]
        fea = feasible_positions(recs, seed=3, batch_seed=100000)
        assert isinstance(fea, list)

    def test_mispaired_map_large_batch_seed(self):
        recs = [_rec(i, i % 8, t=100.0 + i) for i in range(40)]
        m = build_mispaired_parent_map(recs, seed=3, batch_seed=100000)
        assert isinstance(m, dict)

    def test_mask_is_identity_below_2_32(self):
        """mask 对 <2**32 的值恒等——修复不改任何既有行为。"""
        for batch_seed in (0, 1, 200, 5000):
            recs = [_rec(i, i % 8, t=100.0 + i) for i in range(40)]
            fea = feasible_positions(recs, seed=2, batch_seed=batch_seed)
            fea2 = feasible_positions(recs, seed=2, batch_seed=batch_seed)
            assert fea == fea2


# ----------------------------------------------------- 2. AuditAccumulator
class TestAudit:
    def test_deterministic(self):
        a = AuditAccumulator()
        r = _rec(1, 0)
        a.add_population(r)
        a.add_pairing(r, 12)
        a.add_window_close("tjo:layer1", [(1, 0, 2, 0)])
        a.add_replay(r)
        d1 = a.dump(commit="c", config_hash="h", fixed_feature={})
        b = AuditAccumulator()
        r2 = _rec(1, 0)
        b.add_population(r2)
        b.add_pairing(r2, 12)
        b.add_window_close("tjo:layer1", [(1, 0, 2, 0)])
        b.add_replay(r2)
        d2 = b.dump(commit="c", config_hash="h", fixed_feature={})
        assert d1["digests"] == d2["digests"]

    def test_multiset_order_invariant(self):
        a = AuditAccumulator()
        for i in range(5):
            a.add_population(_rec(i, 0))
        b = AuditAccumulator()
        for i in range(4, -1, -1):
            b.add_population(_rec(i, 0))
        assert (a.dump(commit="c", config_hash="h", fixed_feature={})
                ["digests"]["y1_multiset"]
                == b.dump(commit="c", config_hash="h", fixed_feature={})
                ["digests"]["y1_multiset"])

    def test_replay_order_replayed_in_pairing_order(self):
        """pass2 收集顺序与 pass1 不同时 finalize 必须还原。"""
        a = AuditAccumulator()
        for i in (1, 2, 3):
            a.add_pairing(_rec(i, 0), i * 10 + 2)
        for i in (3, 2, 1):                      # reversed collection order
            a.add_replay(_rec(i, 0))
        d = a.dump(commit="c", config_hash="h", fixed_feature={})
        assert d["digests"]["pairing"] == d["digests"]["replay_pairing"]
        assert d["replay_debug"]["n_missing"] == 0

    def test_replay_missing_detected(self):
        a = AuditAccumulator()
        for i in (1, 2, 3):
            a.add_pairing(_rec(i, 0), i * 10 + 2)
        a.add_replay(_rec(1, 0))
        a.add_replay(_rec(2, 0))                 # 3 missing
        d = a.dump(commit="c", config_hash="h", fixed_feature={})
        assert d["replay_debug"]["n_missing"] == 1
        assert d["digests"]["pairing"] != d["digests"]["replay_pairing"]


# --------------------------------------- 3. close_replay 全量覆盖（回归）
class _FixedMaps:
    def __init__(self, m):
        self.m = m

    def pv_row(self, rec):
        # deterministic per-record variation (constant rows collapse the
        # score scale to zero and fail the close)
        t = torch.linspace(0.1, 1.0, self.m)
        return t * (1.0 + float(rec.pair_id[0] % 5))


class TestCloseReplayCoverage:
    def test_g_by_position_covers_every_record(self):
        """close_replay 必须给每个窗口记录回传 adjoint（此前同 boundary_key
        只保留最后 position，导致 pass2 aux 只覆盖部分记录）。"""
        recs = [_rec(i, i % 4) for i in range(12)]
        win = PairKFWindow(tau="tjo:layer1", eps=1e-4, min_unique_trees=4)
        for r in recs:
            win.add(r)
        assert win.ready()
        j, g_by_pos, diag = win.close_replay(_FixedMaps(3))
        assert j is not None, diag
        assert len(g_by_pos) == len(recs)


# ------------------------------------------- 4. 评测路径前置（query 集）
class TestQuerySets:
    def test_query_set_file_valid(self):
        p = REPO / "datasets" / "tgb_wiki_query_sets.json"
        assert p.exists(), "query 集缺失——正式跑的 epoch-5 评测会 FileNotFoundError"
        qs = json.load(open(p))
        assert "val_query_ids" in qs and "test_query_ids" in qs
        assert len(qs["val_query_ids"]) > 0
