"""RPBE/LPSE port onto Meta^n (task book v4.1).

Layout:
  config.py            frozen constants + budget/init assertions
  modes.py             full | official | predictive
  census.py            archive census (run before any training code)
  encoder.py           FrozenItemEncoder: U_v -> X_v, q_emb
  fusion.py            SlottedFusion (Gamma_theta)
  selector.py          PredictiveSelector: A_v -> original Trace/InjectedCode
  records.py           CutRecord
  lineage.py           two-step real lineage extraction
  future.py            phi_r(S_v) fixed sketch
  window.py            StatWindow: B/P/R + OAS CCA
  qp.py                proposal-space QP (ported from fix/avg-lora-clock)
  context_reduction.py full|official|predictive adapter for omega.py
  trainer.py           Phase B offline Gamma training
"""

from meta_n.rpbe.modes import ReductionMode

__all__ = ["ReductionMode"]
