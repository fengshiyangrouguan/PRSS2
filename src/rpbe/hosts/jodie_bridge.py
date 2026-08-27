"""Cut-builder factory for the JODIE line.

Builds the node-grouped, time-sorted FutureIndex over the *full* stream (with
split flags) and wraps it in a :class:`rpbe.records.JodieCutBuilder` for one
training stage: LINK (stage 1, self-supervised pretraining) or NODE_CLASS
(stage 2, node classification).
"""

from rpbe.records import FutureIndex, JodieCutBuilder


def build_cut_builder(dataset, *, stage: str, cfg, seed: int = 0,
                      delta_t_scale: float = 1e6) -> JodieCutBuilder:
    full = dataset.full
    index = FutureIndex(full.sources, full.destinations, full.timestamps,
                        full.labels, val_time=dataset.val_time,
                        test_time=dataset.test_time)
    cfg.delta_t_scale = float(delta_t_scale) if delta_t_scale > 0 else 1.0
    return JodieCutBuilder(index, stage=stage,
                           cuts_per_tau=getattr(cfg, "cuts_per_tau", 32),
                           seed=int(seed))
