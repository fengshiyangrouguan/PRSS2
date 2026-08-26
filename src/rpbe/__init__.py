"""RPBE: predictive recursive compression via a Ky Fan spectral score.

The RPBE component is plugged into the recursive interfaces of a host temporal
model (currently the official twitter-research TGN).  Training adds a single
component-internal loss: the per-interface Ky Fan score
``J_tau = tr[(Sigma_ZZ+eps_z*I)^{-1} Sigma_ZP (Sigma_PP+eps_p*I)^{-1} Sigma_PZ]``
over fixed joint tests ``psi(c,y)`` of real future continuations.  Deployment
keeps only the recursive compressor wrapped around the host aggregate.
"""

from rpbe.compressor import RecursiveCompressor
from rpbe.config import RPBConfig
from rpbe.loss import kf_loss, kf_score
from rpbe.maps import FixedMaps
from rpbe.records import CutRecord

__all__ = ["CutRecord", "FixedMaps", "RPBConfig", "RecursiveCompressor",
           "kf_loss", "kf_score"]

__version__ = "3.0.0"
