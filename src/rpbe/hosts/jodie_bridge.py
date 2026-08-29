"""Leakage-safe cut-builder factory for the JODIE line."""

from rpbe.records import JodieCutBuilder, JodieFutureIndex


def build_cut_builder(dataset, *, stage: str, cfg, seed: int = 0,
                      delta_t_scale: float = 1e6,
                      tables=None, future_index=None) -> JodieCutBuilder:
    """Build from ``dataset.train`` only.

    ``tables`` is retained as a no-op compatibility argument for older
    runners.  Historical edge tables are intentionally never consumed.
    """
    del tables
    if future_index is None:
        stream = dataset.train if hasattr(dataset, "train") else dataset
        future_index = JodieFutureIndex(stream)
    cfg.delta_t_scale = float(delta_t_scale) if delta_t_scale > 0 else 1.0
    return JodieCutBuilder(future_index, stage=stage,
                           cuts_per_tau=getattr(cfg, "cuts_per_tau", 32),
                           seed=int(seed),
                           n_observations=getattr(cfg, "n_observations", 2))
