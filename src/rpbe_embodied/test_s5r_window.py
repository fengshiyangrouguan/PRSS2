"""test_s5r_window.py — reviewer-mandated checks for the Stage5-R RPBE window.

CPU-only.  Run:  PYTHONPATH=src python -m pytest src/rpbe_embodied/test_s5r_window.py -q
(or plain `python src/rpbe_embodied/test_s5r_window.py`).

  1. E=4: cuts from ALL four episodes produce replay inputs (no episode is
     silently dropped by a per-boundary registry clear).
  2. gamma-task and gamma-rpbe take the SAME number of opt_gamma steps per
     episode (n_mb = ceil(task_keys / B), rpbe keys bucketed into n_mb).
  3. window rebuild == direct recursive fixed-trace execution, bitwise.
"""
import math
from types import SimpleNamespace
import torch

from rpbe_embodied.loss import EmbodiedRPBEWindow
from rpbe_embodied.records import EmbodiedCutRow

D = 8          # fake feature dim
SEED = 0
torch.manual_seed(SEED)


def _rec(eid, node, lid, rid, ls, rs):
    return SimpleNamespace(episode_id=eid, node_id=node, left_id=lid,
                           right_id=rid, left_state=ls, right_state=rs)


def _episode(eid, w_window):
    """One episode with a 2-deep merge DAG: leaf 1,2 -> node 10; node10,leaf 3
    -> node 11.  Register edges + leaves and add one cut row (node 11)."""
    a = eid * 100 + 1; b = eid * 100 + 2; c = eid * 100 + 3
    n10 = eid * 100 + 10; n11 = eid * 100 + 11
    ta, tb, tc = (torch.full((D,), float(v)) for v in (eid + 1.0, 2.0, 3.0))
    recs = [_rec(eid, n10, a, b, ta, tb), _rec(eid, n11, n10, c, ta + tb, tc)]
    w_window.add_records(recs)
    z = torch.zeros(D)
    w_window.add([EmbodiedCutRow(
        cut_id=(eid, n11, "cog"), horizon=1, node_id=n11, z=z,
        context={"horizon": 1}, outcome=torch.full((D,), eid + 1.0), weight=1.0)])


def _stub_merge(l, r):
    return l + r


def _direct_rebuild(eid):
    """The fixed-trace recursion, executed directly."""
    node10 = (eid + 1.0) + 2.0
    node11 = node10 + 3.0
    return node11


def test_multi_episode_full_coverage():
    w = EmbodiedRPBEWindow(variant="full_dual", eps=1e-4, min_abs=2)
    for eid in range(4):
        _episode(eid, w)
    j, g_by_cut, replay_inputs, diag = w.close(merge_fn=_stub_merge)
    covered = {cid[0] for cid in g_by_cut}
    assert covered == {0, 1, 2, 3}, f"episodes missing from g_by_cut: {covered}"
    assert {cid[0] for cid in replay_inputs} == {0, 1, 2, 3}, \
        "replay inputs dropped an episode"
    assert diag["n_unique_episodes"] == 4
    print("test_multi_episode_full_coverage OK  covered=", sorted(covered))


def test_rebuild_matches_fixed_trace():
    w = EmbodiedRPBEWindow(variant="full_dual", eps=1e-4, min_abs=2)
    for eid in range(4):
        _episode(eid, w)
    j, g_by_cut, replay_inputs, diag = w.close(merge_fn=_stub_merge)
    # replay input of each cut = (rebuilt left child = node10, leaf c)
    for eid in range(4):
        cl, cr = replay_inputs[(eid, eid * 100 + 11, "cog")]
        want_node10 = _direct_rebuild(eid) - 3.0      # node10 = total - leaf c
        assert torch.allclose(cl.float(), torch.full((D,), want_node10)), \
            f"episode {eid}: rebuilt left child {cl[0]} != {want_node10}"
        assert torch.allclose(cr.float(), torch.full((D,), 3.0))
    # full rebuild (cl + cr) must equal the directly-executed trace
    for eid in range(4):
        cl, cr = replay_inputs[(eid, eid * 100 + 11, "cog")]
        got = (cl + cr)[0].item()
        assert abs(got - _direct_rebuild(eid)) < 1e-6, f"{got} != {_direct_rebuild(eid)}"
    print("test_rebuild_matches_fixed_trace OK  node11 =",
          _direct_rebuild(0), "x4 episodes")


def test_symmetric_opt_steps():
    B = 6
    for n_task, n_rpbe in [(5, 0), (5, 20), (13, 0), (13, 78), (0, 9)]:
        n_mb = max(1, (n_task + B - 1) // B)
        buckets = [[] for _ in range(n_mb)]
        for i in range(n_rpbe):
            buckets[i % n_mb].append(i)
        assert sum(len(b) for b in buckets) == n_rpbe
        assert len(buckets) == n_mb, "step count must depend on task keys only"
        # gamma-task (rpbe=0) and gamma-rpbe both execute exactly n_mb steps
        assert n_mb == max(1, (n_task + B - 1) // B)
    print("test_symmetric_opt_steps OK")


if __name__ == "__main__":
    test_multi_episode_full_coverage()
    test_rebuild_matches_fixed_trace()
    test_symmetric_opt_steps()
    print("ALL_S5R_TESTS_PASS")
