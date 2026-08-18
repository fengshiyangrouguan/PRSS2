"""Pluggable compressor variants and the build factory.

Importing the variant modules registers them into :data:`VARIANT_REGISTRY`.
"""

from prss.compressors.base import (  # noqa: F401
    VARIANT_REGISTRY,
    Compressor,
    InterfaceData,
    build_compressor,
    register_variant,
)
from prss.compressors import (  # noqa: F401  (registration side effects)
    direct,
    pca,
    random,
    spectral,
    vanilla,
)

__all__ = [
    "Compressor",
    "InterfaceData",
    "VARIANT_REGISTRY",
    "build_compressor",
    "register_variant",
]
