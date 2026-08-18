"""Training-only auxiliary objectives: response, spectral tail, unrestricted monitor."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

import torch

from prss.losses import response_loss


@dataclass
class AuxiliaryBatch:
    response_loss: torch.Tensor
    spectral_loss: torch.Tensor
    unrestricted_loss: torch.Tensor
    structured_logits: torch.Tensor
    unrestricted_logits: torch.Tensor
    targets: torch.Tensor
    matrices_by_tau: Dict[str, torch.Tensor]
    occurrence_counts: Dict[str, int]
    contexts_by_tau: Dict[str, torch.Tensor]


def build_auxiliary(prss, trace, root_metadata, root_labels,
                    response_task="binary") -> AuxiliaryBatch:
    """One top-down outside pass over the traced roots.

    Only compressive interfaces (those with a reader) are supervised: d == k base
    interfaces have no quotient to identify and would dominate the auxiliary objective
    through sheer occurrence count.

    Within each traced root, losses are averaged per interface tau first and then
    averaged across represented taus, so a low-level interface cannot dominate merely
    because the computation tree branches.  The spectral Gram itself is still
    accumulated from every valid occurrence within its own tau.

    ``root_metadata`` is a [n_roots, root_metadata_dim] tensor aligned with
    ``trace.root_rows``; the host bridge builds it (it must never contain the label).
    """
    if trace is None or not trace.roots:
        zero = root_labels.sum() * 0.0
        return AuxiliaryBatch(zero, zero, zero, root_labels[:0], root_labels[:0],
                              root_labels[:0], {}, {}, {})

    root_rows = trace.root_rows
    response_root_losses: List[torch.Tensor] = []
    spec_root_losses: List[torch.Tensor] = []
    unres_root_losses: List[torch.Tensor] = []
    all_struct, all_unres, all_targets = [], [], []
    gram_lists = defaultdict(list)
    context_lists = defaultdict(list)
    counts = defaultdict(int)

    row_to_label = {int(r): root_labels[i] for i, r in enumerate(root_rows)}
    row_to_metadata = {int(r): root_metadata[i] for i, r in enumerate(root_rows)}

    for root_oid, top_row in zip(trace.roots, root_rows):
        root = trace.occurrences[root_oid]
        label = row_to_label[int(top_row)]
        metadata = row_to_metadata[int(top_row)]
        contexts = {
            root_oid: prss.outside.root_context(metadata.unsqueeze(0), root.tau).squeeze(0)
        }
        queue = [root_oid]
        order = []
        while queue:
            pid = queue.pop(0)
            order.append(pid)
            parent = trace.occurrences[pid]
            pc = contexts[pid]
            for pos, cid in enumerate(parent.children):
                child = trace.occurrences[cid]
                siblings_by_tau = defaultdict(list)
                for s_pos, sid in enumerate(parent.children):
                    if sid == cid:
                        continue
                    sibling = trace.occurrences[sid]
                    siblings_by_tau[sibling.tau].append(sibling.state.candidate)
                sibling_tensors = {tau: torch.stack(values, dim=0)
                                   for tau, values in siblings_by_tau.items()}
                sib = prss.outside.summarize_siblings(sibling_tensors, pc)
                contexts[cid] = prss.outside.child_context(
                    parent_outside=pc,
                    parent_local=parent.local_features,
                    relation_ids=parent.child_relations[pos],
                    delta_t=parent.child_delta_t[pos],
                    sibling_summary=sib,
                    child_type=child.tau,
                )
                queue.append(cid)

        r_by_tau = defaultdict(list)
        s_by_tau = defaultdict(list)
        u_by_tau = defaultdict(list)
        for oid in order:
            occ = trace.occurrences[oid]
            tau = occ.tau
            if tau not in prss.readers:
                # d == k base interface: no quotient to identify, no reader.
                continue
            c = contexts[oid]
            context_lists[tau].append(c.detach().clone().unsqueeze(0))
            cand = occ.state.candidate.unsqueeze(0)
            y = label.view(1)
            logits, B, _ = prss.structured_read(tau, c.unsqueeze(0), cand)
            ulogit = prss.unrestricted_read(tau, c.detach().unsqueeze(0), cand.detach())
            r_by_tau[tau].append(response_loss(logits, y, task=response_task))
            s_by_tau[tau].append(prss.spectral_loss(tau, B))
            u_by_tau[tau].append(response_loss(ulogit, y, task=response_task))
            # Immutable pre-backward operator snapshot: detach().clone(), never detach()
            # alone (shared storage with the live autograd output).
            gram_lists[tau].append(B.detach().clone())
            counts[tau] += 1
            logit_mon = logits.squeeze(-1) if logits.shape[-1] == 1 else logits
            ulogit_mon = ulogit.squeeze(-1) if ulogit.shape[-1] == 1 else ulogit
            all_struct.append(logit_mon.detach())
            all_unres.append(ulogit_mon.detach())
            all_targets.append(y.detach())

        represented = sorted(r_by_tau)
        if represented:
            response_root_losses.append(
                torch.stack([torch.stack(r_by_tau[tau]).mean() for tau in represented]).mean())
            spec_root_losses.append(
                torch.stack([torch.stack(s_by_tau[tau]).mean() for tau in represented]).mean())
            unres_root_losses.append(
                torch.stack([torch.stack(u_by_tau[tau]).mean() for tau in represented]).mean())

    zero = root_labels.sum() * 0.0
    matrices = {tau: torch.cat(v, dim=0) for tau, v in gram_lists.items() if v}
    contexts_out = {tau: torch.cat(v, dim=0) for tau, v in context_lists.items() if v}
    return AuxiliaryBatch(
        response_loss=(torch.stack(response_root_losses).mean() if response_root_losses else zero),
        spectral_loss=(torch.stack(spec_root_losses).mean() if spec_root_losses else zero),
        unrestricted_loss=(torch.stack(unres_root_losses).mean() if unres_root_losses else zero),
        structured_logits=torch.cat(all_struct) if all_struct else root_labels[:0],
        unrestricted_logits=torch.cat(all_unres) if all_unres else root_labels[:0],
        targets=torch.cat(all_targets) if all_targets else root_labels[:0],
        matrices_by_tau=matrices,
        occurrence_counts=dict(counts),
        contexts_by_tau=contexts_out,
    )
