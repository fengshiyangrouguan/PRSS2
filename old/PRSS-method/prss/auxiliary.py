from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F


@dataclass
class AuxResult:
    response_loss: torch.Tensor
    spectral_loss: torch.Tensor
    unrestricted_loss: torch.Tensor
    structured_logits: torch.Tensor
    unrestricted_logits: torch.Tensor
    targets: torch.Tensor
    matrices_by_layer: Dict[int, torch.Tensor]
    occurrence_counts: Dict[int, int]
    contexts_by_layer: Dict[int, torch.Tensor]


def build_auxiliary(prss, trace, root_rows, root_labels, root_timestamps, log_time_mean, log_time_std):
    """One top-down outside pass over traced source roots.

    Only *compressive recursive interfaces* l>=1 are supervised.  Layer 0 is the
    upstream leaf/base state (d_0 == k_0) and has no quotient to identify; training a
    B(C) reader there both has no spectral meaning and, because leaves outnumber internal
    interfaces combinatorially, would dominate the auxiliary objective.

    Within each traced root, losses are first averaged per recursive layer and then
    averaged across represented layers.  This prevents a lower layer from receiving
    10x/100x more weight solely because the temporal computation tree branches.  The
    spectral Gram itself is still accumulated from every valid occurrence *within its own
    layer*, exactly as the per-interface operator-bank definition requires.
    """
    if trace is None or not trace.roots:
        zero = root_labels.sum() * 0.0
        return AuxResult(zero, zero, zero, root_labels[:0], root_labels[:0], root_labels[:0], {}, {}, {})

    response_root_losses, spec_root_losses, unres_root_losses = [], [], []
    all_struct, all_unres, all_targets = [], [], []
    gram_lists = defaultdict(list)
    context_lists = defaultdict(list)
    counts = defaultdict(int)

    row_to_label = {int(r): root_labels[i] for i, r in enumerate(root_rows)}
    row_to_time = {int(r): root_timestamps[i] for i, r in enumerate(root_rows)}

    for root_oid, top_row in zip(trace.roots, trace.root_rows):
        root = trace.occurrences[root_oid]
        label = row_to_label[int(top_row)]
        ts = row_to_time[int(top_row)]
        norm_logt = (torch.log1p(ts.clamp_min(0)) - log_time_mean) / max(float(log_time_std), 1e-8)
        contexts = {root_oid: prss.outside.root_context(norm_logt, root.layer)}
        queue = [root_oid]
        order = []
        while queue:
            pid = queue.pop(0)
            order.append(pid)
            parent = trace.occurrences[pid]
            pc = contexts[pid]
            for pos, cid in enumerate(parent.children):
                child = trace.occurrences[cid]
                sib = prss.outside.sibling_summary(trace, parent, cid, pc)
                contexts[cid] = prss.outside.child_context(
                    pc, parent.local, parent.relations[pos], parent.deltas[pos], sib, child.layer)
                queue.append(cid)

        r_by_layer = defaultdict(list)
        s_by_layer = defaultdict(list)
        u_by_layer = defaultdict(list)
        for oid in order:
            occ = trace.occurrences[oid]
            # l=0 is a leaf/base state, not a compressive PRSS interface.
            if int(occ.layer) <= 0:
                continue
            c = contexts[oid]
            context_lists[int(occ.layer)].append(c.detach().clone().unsqueeze(0))
            B, b = prss.readers[str(occ.layer)](c.unsqueeze(0))
            cand = occ.candidate.unsqueeze(0)
            y = label.view(1)
            slogit = prss.readers[str(occ.layer)].logits(B, b, cand)
            ulogit = prss.unrestricted[str(occ.layer)](c.detach().unsqueeze(0), cand.detach())
            r_by_layer[int(occ.layer)].append(F.binary_cross_entropy_with_logits(slogit, y))
            s_by_layer[int(occ.layer)].append(prss.quotients[str(occ.layer)].spectral_loss(B))
            u_by_layer[int(occ.layer)].append(F.binary_cross_entropy_with_logits(ulogit, y))
            # Keep an immutable pre-backward operator snapshot. detach() alone shares storage
            # with the live autograd output; the Gram/monitor must never observe later storage reuse.
            gram_lists[int(occ.layer)].append(B.detach().clone())
            counts[int(occ.layer)] += 1
            all_struct.append(slogit.detach())
            all_unres.append(ulogit.detach())
            all_targets.append(y.detach())

        represented = sorted(r_by_layer)
        if represented:
            response_root_losses.append(torch.stack([torch.stack(r_by_layer[l]).mean() for l in represented]).mean())
            spec_root_losses.append(torch.stack([torch.stack(s_by_layer[l]).mean() for l in represented]).mean())
            unres_root_losses.append(torch.stack([torch.stack(u_by_layer[l]).mean() for l in represented]).mean())

    zero = root_labels.sum() * 0.0
    matrices = {l: torch.cat(v, dim=0) for l, v in gram_lists.items() if v}
    contexts_out = {l: torch.cat(v, dim=0) for l, v in context_lists.items() if v}
    return AuxResult(
        response_loss=(torch.stack(response_root_losses).mean() if response_root_losses else zero),
        spectral_loss=(torch.stack(spec_root_losses).mean() if spec_root_losses else zero),
        unrestricted_loss=(torch.stack(unres_root_losses).mean() if unres_root_losses else zero),
        structured_logits=torch.cat(all_struct) if all_struct else root_labels[:0],
        unrestricted_logits=torch.cat(all_unres) if all_unres else root_labels[:0],
        targets=torch.cat(all_targets) if all_targets else root_labels[:0],
        matrices_by_layer=matrices,
        occurrence_counts=dict(counts),
        contexts_by_layer=contexts_out,
    )

