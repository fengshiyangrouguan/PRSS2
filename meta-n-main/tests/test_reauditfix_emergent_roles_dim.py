"""Regression test for re-audit finding #5.

_compute_embeddings previously produced a bare D-dimensional combined_embedding
when only one of {rationale, code} channels was present, while a layer with both
channels got a 2*D vector. _compute_role_differentiation then np.dot()s adjacent
layers' combined_embedding with no shape guard -> ValueError ('shapes not
aligned') when adjacent layers differ in channel presence.

The fix zero-pads the missing channel so every combined_embedding is a fixed
2*D vector. This test builds two adjacent layers with DIFFERING channel presence
(one both-channels, one code-only) and asserts no ValueError plus fixed dims.

Offline: a fake embedding model returns deterministic D-dim vectors; no
sentence-transformers / network / model download required.
"""

import numpy as np

from meta_n.analysis.emergent_roles import EmergentRoleAnalyzer, LayerRoleProfile
from meta_n.core.meta_layer import InjectedCode


_D = 8  # pretend embedding dimensionality


class _FakeModel:
    """Stand-in for a SentenceTransformer: encode(text) -> fixed D-dim vector."""

    def encode(self, text):
        # Deterministic, text-dependent, always dimension _D.
        seed = abs(hash(text)) % (2**32)
        rng = np.random.default_rng(seed)
        return rng.standard_normal(_D).astype(np.float32)


def _analyzer():
    a = EmergentRoleAnalyzer(experiment_dir=".")
    a._embedding_model = _FakeModel()  # bypass lazy load / real model
    return a


def test_adjacent_layers_differing_channel_presence_do_not_crash():
    analyzer = _analyzer()

    # Layer A: both channels present -> pre-fix combined dim = 2*D.
    code_both = InjectedCode(
        pre_process="ctx += 'hint'",
        code_library={"helper": "def helper():\n    return 1"},
        rationale="this layer adds a helper at depth 2",
        source_depth=2,
    )
    # Layer B: code-only, empty rationale -> pre-fix combined dim = D (mismatch!).
    code_only = InjectedCode(
        pre_process=None,
        code_library={"other": "def other():\n    return 2"},
        rationale="",
        source_depth=3,
    )

    profiles = [LayerRoleProfile(depth=2), LayerRoleProfile(depth=3)]
    analyzer._compute_embeddings(profiles, [code_both, code_only])

    # Sanity: layer B genuinely has no rationale channel (the mismatch trigger).
    assert profiles[1].rationale_embedding is None
    assert profiles[1].code_embedding is not None

    # Post-fix: every present combined_embedding is a fixed 2*D vector.
    for p in profiles:
        assert p.combined_embedding is not None
        assert p.combined_embedding.shape == (2 * _D,)

    # The crux: adjacent-pair differentiation must NOT raise ValueError.
    dists = analyzer._compute_role_differentiation(profiles)
    assert "2-3" in dists
    assert isinstance(dists["2-3"], float)
    # Distance is a real number (finite), not nan/inf.
    assert np.isfinite(dists["2-3"])
