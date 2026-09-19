"""Paired / Common-Random-Numbers (CRN) eval — first-cut tests.

Covers (per docs/metan_crn_eval_plan.md):
  * stable_crn_seed: determinism, candidate-independence (compile-time — no
    candidate param exists), repeat/run/task distinctness, NUL-boundary safety,
    a GOLDEN-INT pin (so a regression to builtin hash() fails loudly), in-range.
  * _supports_request_seed: parametrized capability table (default-deny; Azure
    auto-enables; NOT gated behind the temperature predicate).
  * llm_client sink: seed forwarded as int when enabled+supported; DROPPED
    (kwarg absent — byte-identical) when disabled or backend unsupported; exact
    create() keyset for a legacy model when off.
  * orchestrator _crn_seed + eval loop: same (task, repeat) ⇒ same seed across
    two candidates; R repeats ⇒ R distinct seeds elementwise-equal across
    candidates; CRN-off ⇒ _crn_seed is None and no seed reaches the solver.

All tests are LLM-free (mocks only) and assert the DEFAULT-OFF / byte-identity
contract that the 1117-test baseline depends on.
"""

from __future__ import annotations

import inspect
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from meta_n.core.archive import Candidate
from meta_n.core.evolutionary_orchestrator import (
    EvolutionaryConfig,
    EvolutionaryOrchestrator,
)
from meta_n.core.llm_client import (
    LLMClient,
    LLMConfig,
    _supports_request_seed,
    stable_crn_seed,
)
from meta_n.core.meta_layer import TaskDescription, Trace
from meta_n.core.solver import Layer1Solver


# --------------------------------------------------------------------------- #
# 1. stable_crn_seed — pure properties                                        #
# --------------------------------------------------------------------------- #
class TestStableCrnSeed:
    def test_determinism_across_calls(self):
        a = stable_crn_seed(42, "task_a", 0)
        b = stable_crn_seed(42, "task_a", 0)
        assert a == b

    def test_candidate_independence_by_construction(self):
        # The mechanism IS the absence of a candidate_id parameter: parent and
        # child can only call this with (run_seed, task_id, repeat_index), so
        # they are forced onto the same seed. Assert the signature has exactly
        # those three parameters and nothing candidate-shaped.
        params = list(inspect.signature(stable_crn_seed).parameters)
        assert params == ["run_seed", "task_id", "repeat_index"]
        assert not any("candidate" in p.lower() for p in params)

    def test_repeat_distinctness(self):
        # Different repeat_index ⇒ different seed (keeps eval_repeats diverse).
        s0 = stable_crn_seed(42, "task_a", 0)
        s1 = stable_crn_seed(42, "task_a", 1)
        s2 = stable_crn_seed(42, "task_a", 2)
        assert len({s0, s1, s2}) == 3

    def test_run_distinctness(self):
        # Different run_seed ⇒ different seed (the control-independence lever).
        assert stable_crn_seed(42, "task_a", 0) != stable_crn_seed(43, "task_a", 0)
        assert stable_crn_seed(1042, "task_a", 0) != stable_crn_seed(42, "task_a", 0)

    def test_task_distinctness(self):
        assert stable_crn_seed(42, "task_a", 0) != stable_crn_seed(42, "task_b", 0)

    def test_nul_boundary_distinctness(self):
        # The classic boundary collision a NUL separator must prevent:
        # (task="a", r=11) vs (task="a1", r=1) would concatenate to the same
        # bytes without a separator.
        left = stable_crn_seed(42, "a", 11)
        right = stable_crn_seed(42, "a1", 1)
        assert left != right
        # And the symmetric case on the run_seed/task boundary.
        assert stable_crn_seed(1, "1a", 0) != stable_crn_seed(11, "a", 0)

    def test_in_range_31bit_nonnegative(self):
        for rs, tid, ri in [
            (0, "", 0),
            (42, "task_a", 0),
            (2**31, "x" * 200, 9999),
            (123456789, "co_bench/Assortment Problem", 7),
        ]:
            v = stable_crn_seed(rs, tid, ri)
            assert isinstance(v, int)
            assert 0 <= v <= 0x7FFF_FFFF

    def test_golden_int_pins_algorithm(self):
        # GOLDEN PIN: a concrete expected value so a regression to builtin
        # hash() (salted per-process) or a changed digest/mask/separator fails
        # LOUDLY rather than silently breaking --resume and the CO-Bench
        # subprocess workers. Recompute deliberately if the algorithm is ever
        # intentionally changed.
        import hashlib

        run_seed, task_id, repeat_index = 42, "task_a", 0
        basis = f"{run_seed}\x00{task_id}\x00{repeat_index}".encode("utf-8")
        expected = (
            int.from_bytes(hashlib.blake2b(basis, digest_size=8).digest(), "big")
            & 0x7FFF_FFFF
        )
        assert stable_crn_seed(run_seed, task_id, repeat_index) == expected

    def test_int_coercion_of_numeric_args(self):
        # run_seed / repeat_index are int-coerced so a float-typed seed (e.g.
        # from a loosely-typed driver) hashes identically to its int form.
        assert stable_crn_seed(42, "t", 0) == stable_crn_seed(42.0, "t", 0.0)


# --------------------------------------------------------------------------- #
# 2. _supports_request_seed — capability table                               #
# --------------------------------------------------------------------------- #
class TestSupportsRequestSeed:
    @pytest.mark.parametrize(
        "model,backend,expected",
        [
            # Azure auto-enables for every model family, including reasoning
            # families (the whole point: seed is orthogonal to temperature).
            ("gpt-5.2", "azure", True),
            ("gpt-5.2-chat", "azure", True),
            ("gpt-4.1", "azure", True),
            ("o3", "azure", True),
            # OpenRouter / local default-DENY (LM Studio honours seed only at
            # model-load; OpenRouter routing depends on the downstream provider).
            ("anthropic/claude-sonnet-4-20250514", "openrouter", False),
            ("google/gemma-3-27b-it", "openrouter", False),
            ("gpt-5.2", "openrouter", False),
            ("qwen3", "openrouter", False),
            # Unknown backend strings default-deny.
            ("gpt-5.2", "vllm", False),
            ("gpt-5.2", "", False),
        ],
    )
    def test_capability_table(self, model, backend, expected):
        assert _supports_request_seed(model, backend) is expected

    def test_seed_not_gated_behind_temperature_predicate(self):
        # gpt-5.2 rejects a custom temperature (_supports_custom_temperature is
        # False) but MUST still be allowed to carry a seed on Azure — the single
        # highest-value correctness invariant in the plan.
        from meta_n.core.llm_client import _supports_custom_temperature

        assert _supports_custom_temperature("gpt-5.2") is False
        assert _supports_request_seed("gpt-5.2", "azure") is True


# --------------------------------------------------------------------------- #
# 3. llm_client sink — conditional splat / byte-identity                      #
# --------------------------------------------------------------------------- #
def _mock_response(total_tokens: int = 5, content: str = "ok") -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.choices[0].finish_reason = "stop"
    resp.usage = MagicMock()
    resp.usage.prompt_tokens = 2
    resp.usage.completion_tokens = 3
    resp.usage.total_tokens = total_tokens
    resp.usage.prompt_tokens_details = None
    return resp


class TestLLMClientSeedSink:
    @pytest.mark.asyncio
    async def test_seed_forwarded_when_enabled_and_supported(self):
        # Azure backend + a seed value ⇒ seed=int on the wire.
        config = LLMConfig(api_key="k", backend="azure", model="gpt-5.2",
                           azure_endpoint="https://x.openai.azure.com")
        client = LLMClient(config)
        create = AsyncMock(return_value=_mock_response())
        client._client.chat.completions.create = create

        await client.complete([{"role": "user", "content": "hi"}], seed=12345)

        kwargs = create.call_args[1]
        assert kwargs["seed"] == 12345
        assert isinstance(kwargs["seed"], int)

    @pytest.mark.asyncio
    async def test_seed_absent_when_disabled_default_off(self):
        # No seed passed (default path) ⇒ 'seed' kwarg ABSENT (byte-identical),
        # NOT seed=None (which the SDK would serialize as "seed": null).
        config = LLMConfig(api_key="k", backend="azure", model="gpt-5.2",
                           azure_endpoint="https://x.openai.azure.com")
        client = LLMClient(config)
        create = AsyncMock(return_value=_mock_response())
        client._client.chat.completions.create = create

        await client.complete([{"role": "user", "content": "hi"}])

        assert "seed" not in create.call_args[1]

    @pytest.mark.asyncio
    async def test_seed_dropped_for_unsupported_backend(self):
        # OpenRouter backend: even when a seed VALUE is passed, the capability
        # gate drops it → 'seed' kwarg absent → byte-identical.
        config = LLMConfig(api_key="k", model="google/gemma-3-27b-it")  # openrouter
        client = LLMClient(config)
        create = AsyncMock(return_value=_mock_response())
        client._client.chat.completions.create = create

        await client.complete([{"role": "user", "content": "hi"}], seed=999)

        assert "seed" not in create.call_args[1]

    @pytest.mark.asyncio
    async def test_exact_keyset_legacy_model_seed_off(self):
        # The byte-identity guardrail: a legacy OpenRouter model with the feature
        # off sends EXACTLY this keyset — adding the seed plumbing must not
        # introduce or drop any create() kwarg.
        config = LLMConfig(api_key="k", model="anthropic/claude-sonnet-4-20250514")
        client = LLMClient(config)
        create = AsyncMock(return_value=_mock_response())
        client._client.chat.completions.create = create

        await client.complete([{"role": "user", "content": "hi"}])

        assert set(create.call_args[1].keys()) == {
            "model", "messages", "extra_body", "temperature", "max_tokens",
        }

    @pytest.mark.asyncio
    async def test_extra_body_passed_by_identity_when_not_seeding(self):
        # Never mutate / copy the shared self._extra_body when not seeding —
        # pass it through by identity.
        config = LLMConfig(
            api_key="k", model="google/gemma-3-27b-it",
            exclude_providers=["foo"],
        )
        client = LLMClient(config)
        create = AsyncMock(return_value=_mock_response())
        client._client.chat.completions.create = create

        await client.complete([{"role": "user", "content": "hi"}], seed=7)  # dropped

        assert create.call_args[1]["extra_body"] is client._extra_body


# --------------------------------------------------------------------------- #
# 4. Layer1Solver.solve threads seed                                          #
# --------------------------------------------------------------------------- #
class TestSolverThreadsSeed:
    @pytest.mark.asyncio
    async def test_solver_forwards_seed_to_complete(self):
        llm = MagicMock()
        llm.complete = AsyncMock(return_value=("```bash\necho hi\n```", 3))
        solver = Layer1Solver(llm, language="bash")
        await solver.solve(TaskDescription(task_id="t1", description="x"), seed=4242)
        assert llm.complete.await_args.kwargs["seed"] == 4242

    @pytest.mark.asyncio
    async def test_solver_default_seed_is_none(self):
        llm = MagicMock()
        llm.complete = AsyncMock(return_value=("```bash\necho hi\n```", 3))
        solver = Layer1Solver(llm, language="bash")
        await solver.solve(TaskDescription(task_id="t1", description="x"))
        assert llm.complete.await_args.kwargs["seed"] is None


# --------------------------------------------------------------------------- #
# 5. Orchestrator _crn_seed + eval-loop wiring                                #
# --------------------------------------------------------------------------- #
def _orch(**cfg):
    d = dict(max_depth=3, parallel=1, gate_tasks=0)
    d.update(cfg)
    return EvolutionaryOrchestrator(
        llm_client=MagicMock(), executor=MagicMock(), omega=MagicMock(),
        config=EvolutionaryConfig(**d), solver_language="bash",
    )


class TestCrnSeedMethod:
    def test_crn_seed_none_when_paired_eval_off(self):
        orch = _orch(paired_eval=False)  # default
        task = TaskDescription(task_id="t1", description="x")
        assert orch._crn_seed(task, 0) is None
        assert orch._crn_seed(task, 3) is None

    def test_crn_seed_matches_pure_helper_when_on(self):
        orch = _orch(paired_eval=True, seed=42)
        task = TaskDescription(task_id="t1", description="x")
        assert orch._crn_seed(task, 0) == stable_crn_seed(42, "t1", 0)
        assert orch._crn_seed(task, 2) == stable_crn_seed(42, "t1", 2)

    def test_crn_seed_independent_of_candidate(self):
        # Two different candidates, same (task, repeat) ⇒ identical seed.
        orch = _orch(paired_eval=True, seed=42)
        task = TaskDescription(task_id="t1", description="x")
        # _crn_seed takes no candidate argument by design, so any caller for the
        # same (task, repeat) gets the same value.
        assert orch._crn_seed(task, 1) == orch._crn_seed(task, 1)

    def test_crn_seed_honours_run_seed(self):
        task = TaskDescription(task_id="t1", description="x")
        assert _orch(paired_eval=True, seed=42)._crn_seed(task, 0) != (
            _orch(paired_eval=True, seed=1042)._crn_seed(task, 0)
        )


def _capturing_solver(seed_log: list):
    """A solver whose .solve records the seed it was called with."""
    solver = MagicMock()

    async def solve(task, *, seed=None, **_):
        seed_log.append(seed)
        return ("script", "reasoning", 10)

    solver.solve = solve
    return solver


class TestCrnSeedReachesSolver:
    @pytest.mark.asyncio
    async def test_paired_off_no_seed_reaches_solver(self):
        orch = _orch(paired_eval=False)
        seen: list = []
        solver = _capturing_solver(seen)
        orch.executor.execute = AsyncMock(
            return_value=Trace(task_id="t1", success=True, score=0.5, script="s")
        )
        await orch._evaluate_candidate(
            Candidate(candidate_id="c1", depth=1), solver,
            [TaskDescription(task_id="t1", description="x")],
        )
        assert seen == [None]

    @pytest.mark.asyncio
    async def test_paired_on_seed_reaches_solver_r1(self):
        orch = _orch(paired_eval=True, seed=42)
        seen: list = []
        solver = _capturing_solver(seen)
        orch.executor.execute = AsyncMock(
            return_value=Trace(task_id="t1", success=True, score=0.5, script="s")
        )
        await orch._evaluate_candidate(
            Candidate(candidate_id="c1", depth=1), solver,
            [TaskDescription(task_id="t1", description="x")],
        )
        assert seen == [stable_crn_seed(42, "t1", 0)]

    @pytest.mark.asyncio
    async def test_same_task_repeat_same_seed_across_candidates(self):
        # The pairing invariant end-to-end: two candidates, same (task, repeat),
        # see the SAME seed at the solver.
        task = TaskDescription(task_id="t1", description="x")
        seeds_by_cand: dict = {}
        for cid in ("parent", "child"):
            orch = _orch(paired_eval=True, seed=42)
            seen: list = []
            solver = _capturing_solver(seen)
            orch.executor.execute = AsyncMock(
                return_value=Trace(task_id="t1", success=True, score=0.5, script="s")
            )
            await orch._evaluate_candidate(
                Candidate(candidate_id=cid, depth=1), solver, [task]
            )
            seeds_by_cand[cid] = seen
        assert seeds_by_cand["parent"] == seeds_by_cand["child"]
        assert seeds_by_cand["parent"] == [stable_crn_seed(42, "t1", 0)]

    @pytest.mark.asyncio
    async def test_eval_repeats_yields_distinct_seeds_elementwise_equal(self):
        # R repeats ⇒ R DISTINCT seeds (composes with eval_repeats), and the
        # per-repeat seed sequence is elementwise-equal across two candidates.
        task = TaskDescription(task_id="t1", description="x")
        per_cand: dict = {}
        for cid in ("parent", "child"):
            orch = _orch(paired_eval=True, seed=42, eval_repeats=3)
            seen: list = []
            solver = _capturing_solver(seen)
            orch.executor.execute = AsyncMock(
                return_value=Trace(task_id="t1", success=True, score=0.5, script="s")
            )
            await orch._evaluate_candidate(
                Candidate(candidate_id=cid, depth=1), solver, [task]
            )
            per_cand[cid] = seen

        expected = [stable_crn_seed(42, "t1", r) for r in range(3)]
        assert per_cand["parent"] == expected
        assert len(set(expected)) == 3                 # distinct across repeats
        assert per_cand["parent"] == per_cand["child"]  # elementwise-equal

    @pytest.mark.asyncio
    async def test_eval_repeats_off_paired_still_distinct_per_repeat(self):
        # With paired OFF, repeated eval still works and no seed flows.
        orch = _orch(paired_eval=False, eval_repeats=3)
        seen: list = []
        solver = _capturing_solver(seen)
        orch.executor.execute = AsyncMock(
            return_value=Trace(task_id="t1", success=True, score=0.5, script="s")
        )
        await orch._evaluate_candidate(
            Candidate(candidate_id="c1", depth=1), solver,
            [TaskDescription(task_id="t1", description="x")],
        )
        assert seen == [None, None, None]


# --------------------------------------------------------------------------- #
# 5b. Gate solve threads the CRN seed so the reused gate trace IS the draw     #
# --------------------------------------------------------------------------- #
class TestGateSolveCrnSeed:
    def _recording_eval_solve_once(self, seeds: list):
        async def rec(solver, task, candidate, *, seed=None):
            seeds.append(seed)
            return (
                Trace(task_id=task.task_id, depth=1, success=True,
                      score=0.9, script="s"),
                0,
            )
        return rec

    @pytest.mark.asyncio
    async def test_gate_solve_receives_crn_index0_seed_when_on(self):
        # The reused gate trace must be the eval sample-0 (task, index-0) CRN draw
        # — the gate solve now carries stable_crn_seed(seed, task, 0), was None.
        orch = _orch(paired_eval=True, seed=42, gate_tasks=1, gate_repeats=1)
        seeds: list = []
        orch._eval_solve_once = self._recording_eval_solve_once(seeds)
        orch.rng = MagicMock()
        task_a = TaskDescription(task_id="a", description="x")
        orch.rng.sample = lambda pop, k: [task_a]
        child = Candidate(candidate_id="child", parent_id="p", depth=2)
        parent = Candidate(candidate_id="p", depth=1, per_task_scores={"a": 0.5})
        await orch._gate_check(child, parent, MagicMock(), [task_a])
        assert seeds == [stable_crn_seed(42, "a", 0)]

    @pytest.mark.asyncio
    async def test_gate_solve_receives_no_seed_when_off(self):
        # OFF-path byte-identity: _crn_seed is None ⇒ seed=None reaches the solve.
        orch = _orch(paired_eval=False, gate_tasks=1)
        seeds: list = []
        orch._eval_solve_once = self._recording_eval_solve_once(seeds)
        task_a = TaskDescription(task_id="a", description="x")
        cand = Candidate(candidate_id="c", depth=1)
        await orch._gate_solve_one(orch.solver, task_a, cand, repeat_index=0)
        assert seeds == [None]

    @pytest.mark.asyncio
    async def test_topup_solves_receive_per_repeat_crn_seeds_when_on(self):
        # The default topup-ON reuse path: sample 0 is the reused gate trace; the
        # R-1 top-ups carry stable_crn_seed(seed, task, r) for r=1,2 — same basis
        # as the gate sample-0 draw, so the whole median is on one CRN sequence.
        orch = _orch(
            paired_eval=True, seed=42, eval_repeats=3,
            eval_repeats_gate_topup=True,
        )
        seeds: list = []
        orch._eval_solve_once = self._recording_eval_solve_once(seeds)
        task_a = TaskDescription(task_id="a", description="x")
        gate_trace = Trace(task_id="a", depth=1, success=True, score=0.6, script="g")
        cand = Candidate(candidate_id="c", depth=1)
        await orch._evaluate_candidate(
            cand, MagicMock(), [task_a], precomputed={"a": gate_trace},
        )
        assert seeds == [stable_crn_seed(42, "a", 1), stable_crn_seed(42, "a", 2)]

    @pytest.mark.asyncio
    async def test_gate_repeats_get_distinct_per_index_crn_seeds(self):
        # gate_repeats>1: each gate sample gets its OWN repeat_index seed — NOT a
        # shared index-0 seed (which would collapse the median to identical draws).
        orch = _orch(paired_eval=True, seed=42, gate_tasks=1, gate_repeats=2)
        seeds: list = []
        orch._eval_solve_once = self._recording_eval_solve_once(seeds)
        orch.rng = MagicMock()
        task_a = TaskDescription(task_id="a", description="x")
        orch.rng.sample = lambda pop, k: [task_a]
        child = Candidate(candidate_id="child", parent_id="p", depth=2)
        parent = Candidate(candidate_id="p", depth=1, per_task_scores={"a": 0.5})
        await orch._gate_check(child, parent, MagicMock(), [task_a])
        assert seeds == [stable_crn_seed(42, "a", 0), stable_crn_seed(42, "a", 1)]
        assert seeds[0] != seeds[1]


# --------------------------------------------------------------------------- #
# 6. Config field default-off                                                 #
# --------------------------------------------------------------------------- #
class TestConfigField:
    def test_paired_eval_defaults_false(self):
        assert EvolutionaryConfig().paired_eval is False

    def test_paired_eval_round_trips_asdict(self):
        import dataclasses

        cfg = EvolutionaryConfig(paired_eval=True)
        d = dataclasses.asdict(cfg)
        assert d["paired_eval"] is True


# --------------------------------------------------------------------------- #
# 7. L2.1 — paired_eval_effective disclosure (inner channel unseeded)         #
# --------------------------------------------------------------------------- #
class TestPairedEvalDisclosure:
    """L2.1 (audit) — the __init__ disclosure must distinguish three states:

    (a) backend does NOT honour a per-request seed  -> NO-OP warning, effective=False;
    (b) backend DOES honour seed (azure)            -> PARTIALLY-effective warning
        (the inner per-instance llm()/llm_batch() channel is unseeded so reported
        denoising is partial), effective=True;
    (c) default OFF                                  -> no disclosure at all,
        effective=False (byte-identical default path).
    """

    @staticmethod
    def _orch_backend(backend: str, *, paired_eval: bool):
        llm = MagicMock()
        llm.config = SimpleNamespace(model="some-model", backend=backend)
        return EvolutionaryOrchestrator(
            llm_client=llm, executor=MagicMock(), omega=MagicMock(),
            config=EvolutionaryConfig(
                max_depth=3, parallel=1, gate_tasks=0, paired_eval=paired_eval,
            ),
            solver_language="bash",
        )

    def test_unsupported_backend_warns_noop(self, caplog):
        with caplog.at_level(
            logging.WARNING, logger="meta_n.core.evolutionary_orchestrator"
        ):
            orch = self._orch_backend("openrouter", paired_eval=True)
        assert orch._paired_eval_effective is False
        msgs = " ".join(r.getMessage() for r in caplog.records)
        assert "NO-OP" in msgs
        assert "PARTIALLY effective" not in msgs

    def test_azure_backend_warns_partial_inner_channel(self, caplog):
        with caplog.at_level(
            logging.WARNING, logger="meta_n.core.evolutionary_orchestrator"
        ):
            orch = self._orch_backend("azure", paired_eval=True)
        assert orch._paired_eval_effective is True
        msgs = " ".join(r.getMessage() for r in caplog.records)
        assert "PARTIALLY effective" in msgs
        assert "inner" in msgs.lower()
        assert "NO-OP" not in msgs

    def test_default_off_emits_no_disclosure(self, caplog):
        with caplog.at_level(
            logging.WARNING, logger="meta_n.core.evolutionary_orchestrator"
        ):
            orch = self._orch_backend("azure", paired_eval=False)
        assert orch._paired_eval_effective is False
        msgs = " ".join(r.getMessage() for r in caplog.records).lower()
        assert "paired-eval" not in msgs and "paired_eval" not in msgs

    def test_azure_partial_warning_discloses_depth_gt1_gap(self, caplog):
        # R3-D: the PARTIALLY-effective azure warning must ALSO disclose that
        # every bred child (depth>1) gets ZERO CRN denoising in the child-vs-parent
        # SELECTION comparison — the previously-missing caveat on the branch that
        # matters. effective stays True; the NO-OP/PARTIALLY wording is unchanged.
        with caplog.at_level(
            logging.WARNING, logger="meta_n.core.evolutionary_orchestrator"
        ):
            orch = self._orch_backend("azure", paired_eval=True)
        assert orch._paired_eval_effective is True
        msgs = " ".join(r.getMessage() for r in caplog.records)
        assert "depth>1" in msgs                    # the disclosed selection gap
        assert "child" in msgs.lower()              # names the bred-child channel
        assert "PARTIALLY effective" in msgs
        assert "NO-OP" not in msgs
