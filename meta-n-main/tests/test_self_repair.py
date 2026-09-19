"""SelfRepairEvent substrate (Stage 2/3) — round-trip, sidecars, byte-identity.

NON-behavioral substrate. The hard property under test: a candidate that emits
NO ``SelfRepairEvent`` produces byte-identical ``_save_candidate_incremental``
output (no ``repropagation_*`` sidecar), so the all-flags-OFF orchestration is
unchanged from HEAD. Plus the record round-trips and the sidecar writer mirrors
the injected-code sidecar pattern.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from meta_n.core.archive import Candidate
from meta_n.core.evolutionary_orchestrator import (
    EvolutionaryConfig,
    EvolutionaryOrchestrator,
)
from meta_n.core.meta_layer import InjectedCode, Trace
from meta_n.core.self_repair import (
    SELF_REPAIR_SCHEMA_VERSION,
    SelfRepairEvent,
    write_self_repair_sidecars,
)


def _event(**kw) -> SelfRepairEvent:
    base = dict(
        candidate_id="gen2_b0_k0_refined",
        parent_candidate_id="gen2_b0_k0",
        granularity="within_layer",
        target_depth=2,
        pre_code_hash="aaaa1111",
        post_code_hash="bbbb2222",
        mean_before=0.40,
        mean_after=0.62,
        per_task_delta={"task_a": 0.22, "task_b": 0.0},
        accepted=True,
        raw_omega_prompt="REFINE PROMPT BODY",
        raw_omega_response="REFINE RESPONSE BODY",
    )
    base.update(kw)
    return SelfRepairEvent(**base)


class TestSelfRepairEventModel:
    def test_round_trips_through_model_dump(self):
        e = _event()
        e2 = SelfRepairEvent.model_validate(json.loads(e.model_dump_json()))
        assert e2 == e

    def test_defaults_are_inert(self):
        e = SelfRepairEvent()
        assert e.granularity == "within_layer"
        assert e.target_depth == 0
        assert e.accepted is False
        # classification stays None until the Stage-3 classifier fills it.
        assert e.classification is None
        assert e.per_task_delta == {}
        assert e.schema_version == SELF_REPAIR_SCHEMA_VERSION

    def test_sidecar_json_excludes_raw_text(self):
        e = _event()
        d = e.to_sidecar_json()
        assert "raw_omega_prompt" not in d
        assert "raw_omega_response" not in d
        # Everything else survives (mirrors injected_code_d{N}.json).
        assert d["granularity"] == "within_layer"
        assert d["per_task_delta"] == {"task_a": 0.22, "task_b": 0.0}
        assert d["accepted"] is True

    def test_sidecar_text_carries_transcript_and_header(self):
        e = _event()
        txt = e.to_sidecar_text()
        assert "within_layer" in txt
        assert "d2" in txt
        assert "REFINE PROMPT BODY" in txt
        assert "REFINE RESPONSE BODY" in txt


class TestWriteSelfRepairSidecars:
    def test_empty_writes_nothing(self, tmp_path):
        assert write_self_repair_sidecars(tmp_path, []) == []
        assert write_self_repair_sidecars(tmp_path, None) == []
        assert list(tmp_path.iterdir()) == []

    def test_one_event_writes_paired_sidecar(self, tmp_path):
        written = write_self_repair_sidecars(tmp_path, [_event(target_depth=2)])
        names = sorted(p.name for p in written)
        assert names == ["repropagation_d2.json", "repropagation_d2.txt"]
        loaded = json.loads((tmp_path / "repropagation_d2.json").read_text())
        assert loaded["target_depth"] == 2
        assert "raw_omega_prompt" not in loaded
        assert "REFINE RESPONSE BODY" in (tmp_path / "repropagation_d2.txt").read_text()

    def test_collision_same_depth_gets_suffix(self, tmp_path):
        written = write_self_repair_sidecars(
            tmp_path,
            [
                _event(target_depth=2, granularity="within_layer"),
                _event(target_depth=2, granularity="downstream"),
            ],
        )
        names = sorted(p.name for p in written)
        # First event keeps the spec-literal name; the second is suffixed so it
        # does not overwrite the first.
        assert "repropagation_d2.json" in names
        assert "repropagation_d2_1.json" in names

    def test_never_raises_on_bad_dir(self, tmp_path):
        # A path whose parent does not exist -> open() fails -> swallowed.
        bad = tmp_path / "does" / "not" / "exist"
        assert write_self_repair_sidecars(bad, [_event()]) == []


class TestNoEventByteIdentity:
    """A candidate with no events writes NO ``repropagation_*`` file."""

    def _orch(self, tmp_path) -> EvolutionaryOrchestrator:
        config = EvolutionaryConfig(
            output_dir=str(tmp_path),
            max_depth=3, parallel=1, patience=1, gate_tasks=0,
            beam_width=1, beam_candidates=1,
        )
        return EvolutionaryOrchestrator(
            llm_client=MagicMock(),
            executor=MagicMock(),
            omega=MagicMock(),
            config=config,
            solver_language="bash",
        )

    def _candidate(self, cid: str, events=None) -> Candidate:
        return Candidate(
            candidate_id=cid,
            parent_id="gen0_seed",
            iteration=1,
            depth=2,
            injected_codes=[
                InjectedCode(pre_process="additional_context='x'", source_depth=2)
            ],
            traces=[Trace(task_id="task_a", depth=2, script="echo a",
                          success=True, score=0.5)],
            mean_score=0.5,
            per_task_scores={"task_a": 0.5},
            self_repair_events=events or [],
        )

    def test_no_event_candidate_has_no_repropagation_sidecar(self, tmp_path):
        orch = self._orch(tmp_path)
        cand = self._candidate("gen1_b0_k0")
        orch.archive.add(cand)
        orch._save_candidate_incremental(cand, tmp_path)

        cand_dir = tmp_path / "archive" / "gen1_b0_k0"
        files = sorted(p.name for p in cand_dir.iterdir())
        # The default sidecars are present...
        assert "summary.json" in files
        assert "injected_code_d2.json" in files
        # ...and NO self-repair sidecar leaked in.
        assert not any(f.startswith("repropagation_") for f in files)

    def test_event_candidate_emits_sidecar(self, tmp_path):
        orch = self._orch(tmp_path)
        cand = self._candidate(
            "gen1_b0_k0_refined",
            events=[_event(candidate_id="gen1_b0_k0_refined", target_depth=2)],
        )
        orch.archive.add(cand)
        orch._save_candidate_incremental(cand, tmp_path)

        cand_dir = tmp_path / "archive" / "gen1_b0_k0_refined"
        files = sorted(p.name for p in cand_dir.iterdir())
        assert "repropagation_d2.json" in files
        assert "repropagation_d2.txt" in files
        loaded = json.loads((cand_dir / "repropagation_d2.json").read_text())
        assert loaded["candidate_id"] == "gen1_b0_k0_refined"
        assert loaded["accepted"] is True
