# Hotfix v7

Fixes a monitor false-positive hazard: hard finiteness invariants no longer compare a CUDA float mean of a boolean mask against exactly 1.0.  Finiteness is now determined by exact integer non-finite counts.

The PRSS method, reader objective, Gram, and SVD quotient are unchanged.
