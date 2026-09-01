"""LLM-side RPBE pieces (plan v2 L3+).

- ``mem_lift``: J_mem, the fixed structured CountSketch that lifts the SUM
  token K/V memory M_v into the measurement space z_v.
- ``utterance_embed``: frozen utterance embedding (input-embedding mean +
  fixed CountSketch) for the chi features of future observations.
- ``dialogue_records``: the DialogueCutBuilder (one cut -> two horizon
  rows, L4).
"""

from .dialogue_records import DialogueCutBuilder, HORIZON_WEIGHTS
from .mem_lift import JMemLift
from .utterance_embed import UtteranceEmbed

__all__ = ["JMemLift", "UtteranceEmbed", "DialogueCutBuilder",
           "HORIZON_WEIGHTS"]
