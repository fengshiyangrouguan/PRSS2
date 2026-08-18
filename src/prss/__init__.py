"""PRSS: predictive spectral quotient compression for recursive temporal graph hosts.

The core package is host-agnostic: every interface is indexed by an opaque type
string ``tau`` supplied by a host adapter in :mod:`prss.hosts`.  The deployed
recursive state is ``z = R h`` where ``R`` is a per-tau quotient produced by a
pluggable :class:`prss.compressors.Compressor`.
"""

from prss.config import InterfaceSpec, PRSSConfig
from prss.core import PRSSCore

__version__ = "2.0.0"

__all__ = ["PRSSCore", "PRSSConfig", "InterfaceSpec", "__version__"]
