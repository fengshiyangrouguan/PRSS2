"""Cut builder for the JODIE node-classification stage (placeholder).

The real implementation (CutRecord construction from the adapter trace plus a
node-grouped, time-sorted FutureIndex) lands with ``rpbe/records.py``.  This
stub exists so the package imports cleanly in the pure-host build.
"""


class JodieNodeClassificationBridge:
    """Rewritten as the stage-2 CutBuilder in a later step."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "JodieNodeClassificationBridge is implemented in the records step")
