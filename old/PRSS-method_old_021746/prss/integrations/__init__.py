"""Host-model adapters for Predictive Relation-State Sheaf (PRSS)."""

from prss.integrations.tgn import (
  TGNLinkAuxiliaryBatch,
  TGNLinkOutsideBridge,
  TGNNodeAuxiliaryBatch,
  TGNNodeOutsideBridge,
  TGNPreAggregationCandidateLift,
  TGNRecursiveEmbeddingAdapter,
  compute_source_event_embeddings,
  install_tgn_prss,
)

__all__ = [
  "TGNLinkAuxiliaryBatch",
  "TGNLinkOutsideBridge",
  "TGNNodeAuxiliaryBatch",
  "TGNNodeOutsideBridge",
  "TGNPreAggregationCandidateLift",
  "TGNRecursiveEmbeddingAdapter",
  "compute_source_event_embeddings",
  "install_tgn_prss",
]
