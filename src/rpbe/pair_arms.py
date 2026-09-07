"""Structural arms over a shared set of child-parent boundary records.

The three structural arms consume the SAME candidate set (the valid
BoundaryRecords from a trace); they differ ONLY in how the parent future is
used:

* ``1obs``           — child future only (m_p = 0 in the joint map);
* ``2obs_aligned``   — child future + the TRUE parent future (m_p = 1);
* ``2obs_mispaired`` — child future + a permuted parent future (m_p = 1),
                       where the parent futures are deranged within
                       (tau, parent-side/role, coarse parent-time) buckets.

Mispaired invariants (plan step 5):
* receiver's z / child_future / relation metadata are untouched;
* the donor's parent future must be strictly later than the receiver's
  parent time (legal future of the receiving parent);
* the parent-future multiset equals the aligned one exactly (a full
  derangement, every donor used once, none kept in place);
* a bucket with no feasible perfect derangement is dropped from ALL arms, so
  the surviving candidate set stays identical across arms.
"""

import math
from typing import Dict, List, Optional

from rpbe.link_records import BoundaryRecord
from rpbe.records import _fixed_bipartite_derangement


def parent_side(rec: BoundaryRecord) -> int:
    """Which endpoint the parent occurrence is: 0 = src role, 1 = dst role.

    The adapter records a parent as a source query node, but the parent may be
    the source or destination of ITS future event; role distinguishes the
    side for the mispaired bucket (parent-side/role).
    """
    pf = rec.parent_future
    if pf is None:
        return 0
    return int(pf.role)


def _coarse_bucket(times, n_bins=16):
    tmin = min(times) if times else 0.0
    tspan = (max(times) - tmin) if times else 0.0
    if tspan <= 0.0:
        return [0] * len(times)
    return [int((t - tmin) / tspan * n_bins) for t in times]


def _bucket_records(records: List[BoundaryRecord], n_tbins=16):
    """Group records into (tau, parent_side, coarse parent-time) buckets.

    ``n_tbins`` adapts to the pool size so small windows do not fragment every
    bucket to a singleton (which would make the perfect derangement vacuous).
    """
    if not records:
        return {}
    n_eff = max(1, min(int(n_tbins), max(1, len(records) // 3)))
    buckets: Dict[tuple, List[int]] = {}
    parent_times = [r.parent_time for r in records]
    tbins = _coarse_bucket(parent_times, n_eff)
    for i, r in enumerate(records):
        bkey = (r.tau, parent_side(r), tbins[i])
        buckets.setdefault(bkey, []).append(i)
    return buckets


def build_mispaired_parent_map(records: List[BoundaryRecord], *,
                               seed: int, batch_seed: int,
                               n_tbins: int = 16) -> Dict[int, object]:
    """Perfect-derange parent futures among records; drop infeasible buckets.

    Returns ``{position: parent_future}`` mapping a record position in
    ``records`` to the ObservedLinkEvent that position should use.  Records
    in an infeasible bucket are ABSENT from the map (caller drops them from
    every arm).  Deterministic in (seed, batch_seed, bucket contents).
    """
    buckets = _bucket_records(records, n_tbins)
    mapping: Dict[int, object] = {}
    for bkey, idxs in buckets.items():
        if len(idxs) < 2:
            continue
        # donor parent future must be a legal future of the receiver's parent
        times_from = [records[i].parent_time for i in idxs]
        times_to = [records[i].parent_future.time for i in idxs]
        bucket_seed = 0
        for i in idxs:
            bucket_seed = (bucket_seed * 31 + i) % (2 ** 31)
        perm = _fixed_bipartite_derangement(
            times_from, times_to,
            (seed * 104729) ^ int(batch_seed) ^ bucket_seed)
        if perm is None:
            # infeasible bucket: drop from ALL arms (absent from mapping)
            continue
        for pos, i in enumerate(idxs):
            donor_idx = idxs[perm[pos]]
            assert donor_idx != i, "derangement must move every parent future"
            mapping[i] = records[donor_idx].parent_future
    return mapping


def feasible_positions(records: List[BoundaryRecord], *,
                       seed: int, batch_seed: int,
                       n_tbins: int = 16):
    """Positions of records that survive the mispaired feasibility drop.

    Used by ALL arms so the surviving candidate set is identical.
    """
    buckets = _bucket_records(records, n_tbins)
    ok = set()
    for bkey, idxs in buckets.items():
        if len(idxs) < 2:
            continue
        times_from = [records[i].parent_time for i in idxs]
        times_to = [records[i].parent_future.time for i in idxs]
        bucket_seed = 0
        for i in idxs:
            bucket_seed = (bucket_seed * 31 + i) % (2 ** 31)
        perm = _fixed_bipartite_derangement(
            times_from, times_to,
            (seed * 104729) ^ int(batch_seed) ^ bucket_seed)
        if perm is not None:
            ok.update(idxs)
    return sorted(ok)
