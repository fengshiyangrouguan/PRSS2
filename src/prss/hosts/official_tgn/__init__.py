"""Vendored upstream twitter-research/tgn (commit d55bbe678acabb9fc3879c408fd1f2e15919667c).

Only the import lines were rewritten to package paths; every other line is
byte-for-byte identical to upstream (verified via UPSTREAM_CORE_SHA256.json,
see README.md).  This module is used by the JODIE-protocol training line.
"""

from prss.hosts.official_tgn.model.tgn import TGN
from prss.hosts.official_tgn.utils.utils import MLP, NeighborFinder, get_neighbor_finder

__all__ = ["TGN", "MLP", "NeighborFinder", "get_neighbor_finder"]
