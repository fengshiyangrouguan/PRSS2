"""Proper response and normalized predictive spectral-tail losses."""

import torch
from torch.nn import functional as F


def response_loss(logits, target, task="binary", reduction="mean"):
    if task == "binary":
        if logits.shape[-1] != 1:
            raise ValueError("Binary response reader must have response_dim=1")
        target = target.to(dtype=logits.dtype).reshape(logits.shape[:-1])
        return F.binary_cross_entropy_with_logits(logits.squeeze(-1), target,
                                                  reduction=reduction)
    if task == "multiclass":
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1).long(),
                               reduction=reduction)
    if task == "gaussian_mean":
        target = target.to(dtype=logits.dtype).reshape_as(logits)
        return F.mse_loss(logits, target, reduction=reduction)
    if task == "huber":
        target = target.to(dtype=logits.dtype).reshape_as(logits)
        return F.huber_loss(logits, target, reduction=reduction)
    raise ValueError("Unknown response task: {}".format(task))


def spectral_tail_per_occurrence(reader_matrix, quotient_rows, eps=1e-8):
    """||B(I-P_R)||_F^2 / (||B||_F^2 + eps), with R detached."""
    if reader_matrix.shape[-1] != quotient_rows.shape[-1]:
        raise ValueError("Reader and quotient candidate dimensions differ")
    rows = quotient_rows.detach()
    projector = rows.transpose(-1, -2) @ rows
    identity = torch.eye(projector.shape[-1], device=projector.device,
                         dtype=projector.dtype)
    residual = reader_matrix @ (identity - projector)
    numerator = residual.square().sum(dim=(-2, -1))
    denominator = reader_matrix.square().sum(dim=(-2, -1)) + eps
    return numerator / denominator


def spectral_tail_loss(reader_matrix, quotient_rows, eps=1e-8, reduction="mean"):
    values = spectral_tail_per_occurrence(reader_matrix, quotient_rows, eps=eps)
    if reduction == "none":
        return values
    if reduction == "sum":
        return values.sum()
    if reduction == "mean":
        return values.mean()
    raise ValueError("Unknown reduction: {}".format(reduction))


def predictive_state_tail_loss(reader_matrix, candidate, quotient_rows, eps=1e-8,
                               reduction="mean"):
    """Predictive response lost by projecting one candidate through the current quotient.

    Unlike ``spectral_tail_loss(B, R)``, this loss detaches the learned reader matrix B
    and R and backpropagates only into the candidate representation, so the reader cannot
    self-confirm an arbitrary current subspace.
    """
    if reader_matrix.shape[-1] != candidate.shape[-1]:
        raise ValueError("Reader/candidate dimensions differ")
    rows = quotient_rows.detach()
    matrix = reader_matrix.detach()
    projector = rows.transpose(-1, -2) @ rows
    projected = candidate @ projector
    discarded = candidate - projected
    lost_response = torch.einsum("...pd,...d->...p", matrix, discarded)
    full_response = torch.einsum("...pd,...d->...p", matrix, candidate)
    numerator = lost_response.square().sum(dim=-1)
    scale = matrix.square().sum(dim=(-2, -1)) * candidate.square().sum(dim=-1)
    denominator = full_response.square().sum(dim=-1) + 0.01 * scale + eps
    values = numerator / denominator
    if reduction == "none":
        return values
    if reduction == "sum":
        return values.sum()
    if reduction == "mean":
        return values.mean()
    raise ValueError("Unknown reduction: {}".format(reduction))
