"""§6b F181 — SelfRepairEvent reserved sidecar refs: schema-v2 shape pin.

At schema v2, ``pre_injection_ref`` / ``omega_prompt_ref`` /
``omega_response_ref`` are RESERVED: they always serialize as ``null`` (only
``post_injection_ref`` is populated, by the orchestrator at event
construction). A ``null`` ref means "not wired yet", never "transcript absent"
— the transcript lives at the sidecar's paired ``{stem}.txt`` by naming
convention. These tests pin that contract so populating a reserved ref without
bumping ``SELF_REPAIR_SCHEMA_VERSION`` trips a test.
"""

from meta_n.core.self_repair import SELF_REPAIR_SCHEMA_VERSION, SelfRepairEvent


def _event() -> SelfRepairEvent:
    return SelfRepairEvent(
        candidate_id="cand_new",
        parent_candidate_id="cand_parent",
        granularity="downstream",
        target_depth=3,
        pre_code_hash="aaaa1111",
        post_code_hash="bbbb2222",
        post_injection_ref="injected_code_d3.json",
        mean_before=0.4,
        mean_after=0.6,
        accepted=True,
        archived=True,
        raw_omega_prompt="PROMPT",
        raw_omega_response="RESPONSE",
    )


class TestReservedRefsSchemaV2:
    def test_reserved_refs_null_at_schema_v2(self):
        data = _event().to_sidecar_json()

        # Reserved refs: PRESENT in the sidecar, value null.
        for key in ("pre_injection_ref", "omega_prompt_ref", "omega_response_ref"):
            assert key in data
            assert data[key] is None

        # The one populated ref, and the version stamp the contract hangs on.
        assert data["post_injection_ref"] == "injected_code_d3.json"
        assert data["schema_version"] == SELF_REPAIR_SCHEMA_VERSION == 2

    def test_reserved_refs_roundtrip_through_archive_loader_shape(self):
        """Mirrors the archive-resume path: ``model_validate`` over the sidecar
        dict (archive.py round-trips ``repropagation_d*.json`` this way) must
        preserve the reserved nulls, not coerce or drop them."""
        loaded = SelfRepairEvent.model_validate(_event().to_sidecar_json())

        assert loaded.pre_injection_ref is None
        assert loaded.omega_prompt_ref is None
        assert loaded.omega_response_ref is None
        assert loaded.post_injection_ref == "injected_code_d3.json"
        assert loaded.schema_version == SELF_REPAIR_SCHEMA_VERSION
