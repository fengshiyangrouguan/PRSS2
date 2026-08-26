"""Validation/test isolation audit for the RPBE component.

The new "three zeros": during evaluation nothing may build a computation-tree
trace, the fixed measurement maps (buffers / seed / normalization) must not be
touched, and no training-state flag may be flipped back.  ``assert_clean``
hard-fails on any violation.
"""

from typing import Dict, Optional


def rpbe_fingerprint(fixed_maps) -> Optional[Dict]:
    if fixed_maps is None:
        return None
    return fixed_maps.isolation_fingerprint()


def assert_clean(before: Optional[Dict], fixed_maps, trace_created: bool,
                 label: str) -> None:
    """Compare the pre-evaluation fingerprint with the post-evaluation one."""
    if fixed_maps is None:
        assert not trace_created, (
            "{}: a trace was built without a fixed-measurement component"
            .format(label))
        return
    after = rpbe_fingerprint(fixed_maps)
    if before is None:
        raise RuntimeError("{}: no pre-evaluation fingerprint captured"
                           .format(label))
    if trace_created:
        raise RuntimeError(
            "{}: computation-tree trace built during evaluation".format(label))
    if after != before:
        raise RuntimeError(
            "{}: fixed measurement maps changed during evaluation".format(label))
