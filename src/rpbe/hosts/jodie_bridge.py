"""Cut-builder factory for the JODIE line.

Builds the explicit ``idx -> endpoints`` / ``idx -> label`` tables over the
*full* stream (no indexing-convention assumptions) and wraps them in a
:class:`rpbe.records.JodieCutBuilder` for one training stage: LINK (stage 1,
self-supervised pretraining) or NODE_CLASS (stage 2, node classification).
The same tables feed the adapter's consumption-record stamping.
"""

from rpbe.records import build_edge_tables, JodieCutBuilder


def build_cut_builder(dataset, *, stage: str, cfg, seed: int = 0,
                      delta_t_scale: float = 1e6,
                      tables=None) -> JodieCutBuilder:
    if tables is None:
        (endpoints, labels, users, pages,
         _) = build_edge_tables(dataset)
        tables = (endpoints, labels, users, pages)
    cfg.delta_t_scale = float(delta_t_scale) if delta_t_scale > 0 else 1.0
    return JodieCutBuilder(tables, stage=stage,
                           cuts_per_tau=getattr(cfg, "cuts_per_tau", 32),
                           seed=int(seed))
