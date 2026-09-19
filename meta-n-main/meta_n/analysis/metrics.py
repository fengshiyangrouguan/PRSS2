"""Post-hoc metrics analysis for Meta^n experiments.

Computes five metric categories from saved experiment artifacts:
1. Convergence & Efficiency
2. Emergent Abstraction
3. Archive Diversity
4. Failure Pattern Reduction
5. Robustness

Usage:
    python -m meta_n.analysis.metrics experiments/full_cobench_evo_B2K2/
    python -m meta_n.analysis.metrics experiments/pilot_cobench_10tasks/ --no-plots
"""

from __future__ import annotations

import ast as ast_mod
import json
import logging
import math
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Optional

from meta_n.core.meta_layer import InjectedCode, Trace, classify_error
from meta_n.core.omega import TASK_CATEGORIES, OmegaEngine

logger = logging.getLogger(__name__)


def dev_test_direction(gap: float, eps: float = 0.02) -> str:
    """Classify a dev−test gap (6.4a). The SIGN is property-dependent, NOT a
    universal overfit signal: ``dev>test`` (positive) = overfit; ``test>dev``
    (negative, e.g. CO-Bench-gpt5.2) = the dev split was pessimistic and the
    system GENERALIZES (GOOD, never overfit); within ``eps`` = matched."""
    return "overfit" if gap > eps else "generalizes" if gap < -eps else "matched"


def _load_contiguous_injected_codes(
    path_for_depth: Callable[[int], Path],
    start_depth: int = 2,
    augment: Optional[Callable[[int, dict], None]] = None,
) -> list[InjectedCode]:
    """Load ``InjectedCode`` blobs for depth = start_depth, start_depth+1, ...

    Contiguous while-loop semantics: stops at the FIRST missing depth (a gap
    ends the chain — deliberately NOT a glob, which would tolerate gaps).
    ``augment(depth, data)``, when given, may mutate the parsed dict before
    ``InjectedCode(**data)`` is constructed (e.g. merging a raw omega response).
    Shared by the archive and linear layouts here and by
    :meth:`meta_n.analysis.emergent_roles.EmergentRoleAnalyzer._load_injected_codes`.
    """
    codes: list[InjectedCode] = []
    depth = start_depth
    while True:
        path = path_for_depth(depth)
        if not path.exists():
            break
        with open(path) as f:
            data = json.load(f)
        if augment is not None:
            augment(depth, data)
        codes.append(InjectedCode(**data))
        depth += 1
    return codes


def _iteration_of(c: dict) -> int:
    """Iteration of an archive-index candidate row (legacy rows say ``generation``)."""
    return c.get("iteration", c.get("generation", 0))


def _best_candidate_by_iteration(candidates: list[dict]) -> dict[int, dict]:
    """Map iteration → the highest-``mean_score`` candidate row of that iteration.

    Strict ``>`` keeps the FIRST-seen candidate on ties; the tie-breaking is
    part of the persisted metrics.json contract and must not change.
    """
    by_iter: dict[int, dict] = {}
    for c in candidates:
        it = _iteration_of(c)
        if it not in by_iter or c.get("mean_score", 0) > by_iter[it].get("mean_score", 0):
            by_iter[it] = c
    return by_iter


class ExperimentAnalyzer:
    """Post-hoc metrics analysis for Meta^n experiments."""

    def __init__(
        self,
        experiment_dir: str | Path,
        embedding_model: str = "all-MiniLM-L6-v2",
        use_embeddings: bool = True,
    ):
        self.experiment_dir = Path(experiment_dir)
        self.mode = self._detect_mode()
        self._embedding_model_name = embedding_model
        self._use_embeddings = use_embeddings
        self._embedding_model = None
        # Cache loaded data
        self._summary: dict | None = None
        self._archive_index: dict | None = None

    def _detect_mode(self) -> str:
        if (self.experiment_dir / "archive" / "index.json").exists():
            return "evolutionary"
        if (self.experiment_dir / "depth_2").exists():
            return "linear"
        # External-baseline runs (e.g. baselines/godel_agent) emit a Meta^n-
        # compatible summary.json with `archive_semantics: <something>` so we
        # can recognise them without an archive/ tree or depth_N/ folders.
        summary_path = self.experiment_dir / "summary.json"
        if summary_path.exists():
            try:
                with open(summary_path) as f:
                    summary = json.load(f)
                if summary.get("archive_semantics"):
                    return "baseline"
            except Exception:
                pass
        raise FileNotFoundError(
            f"Cannot detect experiment mode in {self.experiment_dir}"
        )

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_summary(self) -> dict:
        if self._summary is None:
            with open(self.experiment_dir / "summary.json") as f:
                self._summary = json.load(f)
        return self._summary

    def _load_config(self) -> dict:
        path = self.experiment_dir / "config.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {}

    def _load_archive_index(self) -> dict | None:
        if self._archive_index is not None:
            return self._archive_index
        path = self.experiment_dir / "archive" / "index.json"
        if path.exists():
            with open(path) as f:
                self._archive_index = json.load(f)
            return self._archive_index
        return None

    def _load_convergence_history(self) -> list[float]:
        summary = self._load_summary()
        if "convergence_history" in summary:
            return summary["convergence_history"]
        # Linear mode: reconstruct from mean_scores
        mean_scores = summary.get("mean_scores", {})
        return [mean_scores[str(d)] for d in sorted(int(k) for k in mean_scores)]

    def _load_candidate_traces(self, candidate_id: str) -> list[dict]:
        traces_dir = self.experiment_dir / "archive" / candidate_id / "traces"
        if not traces_dir.exists():
            return []
        traces = []
        for f in sorted(traces_dir.glob("*.json")):
            with open(f) as fp:
                traces.append(json.load(fp))
        return traces

    def _load_depth_traces(self, depth: int) -> list[dict]:
        traces_dir = self.experiment_dir / f"depth_{depth}" / "traces"
        if not traces_dir.exists():
            return []
        traces = []
        for f in sorted(traces_dir.glob("*.json")):
            with open(f) as fp:
                traces.append(json.load(fp))
        return traces

    def _load_injected_codes_for_candidate(self, candidate_id: str) -> list[InjectedCode]:
        cand_dir = self.experiment_dir / "archive" / candidate_id
        return _load_contiguous_injected_codes(
            lambda d: cand_dir / f"injected_code_d{d}.json"
        )

    def _load_injected_codes_linear(self) -> list[InjectedCode]:
        return _load_contiguous_injected_codes(
            lambda d: self.experiment_dir / f"depth_{d}" / "injected_code.json"
        )

    def _load_best_chain_codes(self) -> list[InjectedCode]:
        if self.mode == "evolutionary":
            lineage_path = self.experiment_dir / "lineage" / "best_chain.json"
            if lineage_path.exists():
                with open(lineage_path) as f:
                    lineage = json.load(f)
                best_id = lineage.get("best_candidate_id", "")
                return self._load_injected_codes_for_candidate(best_id)
        return self._load_injected_codes_linear()

    def _load_test_results(self) -> dict | None:
        path = self.experiment_dir / "test_results.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None

    def _get_seed_traces(self) -> list[dict]:
        if self.mode == "evolutionary":
            return self._load_candidate_traces("gen0_seed")
        return self._load_depth_traces(1)

    def _get_best_candidate_traces(self) -> list[dict]:
        if self.mode == "evolutionary":
            summary = self._load_summary()
            best_id = summary.get("best_candidate_id", "")
            if best_id:
                return self._load_candidate_traces(best_id)
        # Linear: use the depth with highest mean_score (not necessarily final)
        summary = self._load_summary()
        mean_scores = summary.get("mean_scores", {})
        if mean_scores:
            best_depth = max(mean_scores, key=lambda d: mean_scores[d])
            return self._load_depth_traces(int(best_depth))
        final_depth = summary.get("final_depth", 1)
        return self._load_depth_traces(final_depth)

    def _get_embedding_model(self):
        # Degrades to None on ImportError (callers check for a None model). The
        # emergent_roles.EmergentRoleAnalyzer._get_embedding_model twin has a
        # deliberately different error contract (raises; its caller catches) —
        # do not consolidate.
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer(self._embedding_model_name)
            except ImportError:
                logger.info("sentence-transformers not installed, embeddings unavailable")
        return self._embedding_model

    # ------------------------------------------------------------------
    # Metric 1: Convergence & Efficiency
    # ------------------------------------------------------------------

    def compute_convergence(self) -> dict:
        """Compute convergence trajectory and efficiency metrics."""
        history = self._load_convergence_history()
        summary = self._load_summary()

        # Marginal improvement
        marginal = [None]
        for i in range(1, len(history)):
            marginal.append(history[i] - history[i - 1])

        # Cumulative tokens
        total_tokens = summary.get("total_tokens", 0)
        if self.mode == "evolutionary":
            index = self._load_archive_index()
            candidates = index.get("candidates", []) if index else []
            # Group by iteration, sum tokens
            iter_tokens: dict[int, int] = {}
            for c in candidates:
                it = _iteration_of(c)
                iter_tokens[it] = iter_tokens.get(it, 0) + c.get("total_tokens", 0)
            cumulative = []
            running = 0
            for i in range(len(history)):
                running += iter_tokens.get(i, 0)
                cumulative.append(running)
        else:
            tokens_per_depth = summary.get("tokens_per_depth", {})
            cumulative = []
            running = 0
            for i in range(len(history)):
                running += tokens_per_depth.get(str(i + 1), 0)
                cumulative.append(running)

        # Tokens per improvement
        tpi = [None]
        for i in range(1, len(history)):
            delta = marginal[i]
            step_tokens = cumulative[i] - cumulative[i - 1]
            if delta and delta > 0:
                tpi.append(step_tokens / delta)
            else:
                tpi.append(None)

        # Time to thresholds
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        time_to = {}
        for t in thresholds:
            idx = next((i for i, s in enumerate(history) if s >= t), None)
            time_to[str(t)] = idx

        return {
            "convergence_history": history,
            "marginal_improvement": marginal,
            "cumulative_tokens": cumulative,
            "tokens_per_improvement": tpi,
            "time_to_thresholds": time_to,
            "total_tokens": total_tokens,
            "final_score": history[-1] if history else 0.0,
            "total_steps": len(history),
        }

    # ------------------------------------------------------------------
    # Metric 2: Emergent Abstraction
    # ------------------------------------------------------------------

    def compute_emergent_abstraction(self) -> dict:
        """Analyze abstraction gradient across depths in the best chain."""
        codes = self._load_best_chain_codes()
        if not codes:
            return {"skipped": "no injected codes found"}

        # Abstraction gradient
        abstraction_gradient = []
        for ic in codes:
            score = self._compute_abstraction_score(ic)
            abstraction_gradient.append(score)

        # Improvement type profiles
        type_profiles = []
        for i, ic in enumerate(codes):
            scores = self._classify_improvement_types(ic)
            top = sorted(scores.items(), key=lambda x: -x[1])[:3]
            type_profiles.append({
                "depth": i + 2,
                "top_types": [t[0] for t in top if t[1] > 0.2],
                "scores": scores,
            })

        # Code complexity gradient
        complexity_gradient = []
        for i, ic in enumerate(codes):
            # T3.7: fold the Python code_library channel (helper VALUES) into the
            # AST input so a pure-helper layer is not reported zero-complexity.
            # Bash helpers are not valid Python and are excluded from the AST.
            code = "\n".join(filter(None, [ic.pre_process, *ic.code_library.values()]))
            complexity = self._compute_ast_complexity(code)
            complexity["depth"] = i + 2
            complexity_gradient.append(complexity)

        # Vocabulary shift (embedding distance between adjacent rationales)
        vocab_shift = {}
        if self._use_embeddings and len(codes) >= 2:
            model = self._get_embedding_model()
            if model:
                rationales = [ic.rationale or "" for ic in codes]
                embeddings = model.encode(rationales)
                import numpy as np
                for i in range(len(embeddings) - 1):
                    a, b = embeddings[i], embeddings[i + 1]
                    cos_dist = 1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
                    vocab_shift[f"{i+2}-{i+3}"] = round(cos_dist, 4)

        return {
            "best_chain": {
                "depth": len(codes) + 1,
                "abstraction_gradient": abstraction_gradient,
                "improvement_type_profiles": type_profiles,
                "code_complexity_gradient": complexity_gradient,
                "vocabulary_shift": vocab_shift,
            }
        }

    @staticmethod
    def _compute_abstraction_score(ic: InjectedCode) -> float:
        """Score 0 (task-specific) to 1 (generic). Based on pattern ratios.

        Primary-pipeline heuristic — DELIBERATELY different from
        ``emergent_roles.EmergentRoleAnalyzer._compute_abstraction_score`` (the
        Exp-6 variant uses different keyword lists plus per-task 4-gram
        literal-phrase matching). Scores are numerically incomparable across
        the two files: this one feeds metrics.json's byte-pinned
        ``best_chain.abstraction_gradient``; the Exp-6 outputs serialize with
        an ``_exp6`` suffix in role_analysis.json. Do not consolidate;
        T3.6/T3.7-class channel-fold changes must be applied to both copies in
        parallel.
        """
        code = ic.pre_process or ""
        rationale = ic.rationale or ""
        # T3.7: include the code_library channel (helper VALUES) so a pure-helper
        # layer's abstraction is measured over its actual code, not just prompts.
        helpers = "\n".join(
            list(ic.code_library.values()) + list(ic.code_library_bash.values())
        )
        text = code + " " + rationale + " " + helpers

        generic_patterns = [
            "task.description", "task.task_id", "task_id", "isinstance",
            "category", "classify", "strategy", "systemic", "structural",
            "pattern", "framework", "policy",
        ]
        specific_patterns = [
            "scipy", "numpy", "timeout", "error", "import", "module",
            "index", "boundary", "overlap",
        ]

        generic_count = sum(1 for p in generic_patterns if p in text.lower())
        specific_count = sum(1 for p in specific_patterns if p in text.lower())
        total = generic_count + specific_count
        if total == 0:
            return 0.5
        return round(generic_count / total, 3)

    @staticmethod
    def _classify_improvement_types(ic: InjectedCode) -> dict[str, float]:
        """Classify improvement type using keyword matching.

        Primary-pipeline heuristic — DELIBERATELY different from
        ``emergent_roles.EmergentRoleAnalyzer._classify_improvement_type`` (the
        Exp-6 variant scores 7 enum types with AST signals and
        max-normalization; this one scores 5 types with
        ``min(count/(len*0.5), 1)``). Scores are numerically incomparable
        across the two files (the Exp-6 outputs serialize with an ``_exp6``
        suffix). Do not consolidate; T3.6/T3.7-class channel-fold changes must
        be applied to both copies in parallel.
        """
        type_keywords = {
            "prompt_mod": ["hint", "context", "prompt", "instruction", "guidance", "additional_context"],
            "strategy_selection": ["classify", "category", "strategy", "select", "choose", "detect", "route"],
            "error_handling": ["error", "handle", "catch", "recover", "exception", "try", "validate"],
            "control_flow": ["conditional", "branch", "if ", "elif ", "switch"],
            "decomposition": ["decompose", "break down", "sub-task", "step", "split", "phase"],
        }
        code = ic.pre_process or ""
        rationale = ic.rationale or ""
        # T3.7: include the code_library channel (helper VALUES) so a pure-helper
        # layer is classified over its actual code, not just the prompt channel.
        helpers = "\n".join(
            list(ic.code_library.values()) + list(ic.code_library_bash.values())
        )
        text = (code + " " + rationale + " " + helpers).lower()

        scores = {}
        for typ, keywords in type_keywords.items():
            count = sum(1 for kw in keywords if kw in text)
            scores[typ] = min(count / max(len(keywords) * 0.5, 1), 1.0)
        return {k: round(v, 3) for k, v in scores.items()}

    @staticmethod
    def _compute_ast_complexity(code: str) -> dict:
        """Compute AST-based complexity metrics."""
        if not code or not code.strip():
            return {"num_nodes": 0, "max_depth": 0, "num_branches": 0, "num_functions": 0}
        try:
            tree = ast_mod.parse(code)
        except SyntaxError:
            return {"num_nodes": 0, "max_depth": 0, "num_branches": 0, "num_functions": 0}

        num_nodes = sum(1 for _ in ast_mod.walk(tree))
        num_branches = sum(
            1 for n in ast_mod.walk(tree)
            if isinstance(n, (ast_mod.If, ast_mod.For, ast_mod.While, ast_mod.Match))
        )
        num_functions = sum(
            1 for n in ast_mod.walk(tree)
            if isinstance(n, (ast_mod.FunctionDef, ast_mod.AsyncFunctionDef))
        )

        # Max nesting depth
        def _depth(node, current=0):
            children = list(ast_mod.iter_child_nodes(node))
            if not children:
                return current
            return max(_depth(c, current + 1) for c in children)

        max_depth = _depth(tree)

        return {
            "num_nodes": num_nodes,
            "max_depth": max_depth,
            "num_branches": num_branches,
            "num_functions": num_functions,
        }

    # ------------------------------------------------------------------
    # Metric 3: Archive Diversity
    # ------------------------------------------------------------------

    def compute_archive_diversity(self) -> dict:
        """Compute diversity metrics for the archive (evolutionary only)."""
        if self.mode != "evolutionary":
            # Renders byte-identically for linear runs; baseline runs are
            # labelled with their own mode rather than mislabelled "linear".
            return {"skipped": f"{self.mode} mode — no archive"}

        summary = self._load_summary()
        index = self._load_archive_index()
        if not index:
            return {"skipped": "no archive index"}

        ptb = index.get("per_task_best", {})
        best_mean = summary.get("best_mean_score", 0.0)

        # Oracle gap. per_task_best omits tasks with no finite-scored candidate
        # (Archive.per_task_best_scores contract), so the bare-subset
        # denominator inflates the oracle — same Audit-#17 class the
        # run_persistence / orchestrator writers already fixed.
        oracle_scores = {tid: entry["score"] for tid, entry in ptb.items()}
        oracle_note = None
        oracle_mean = self._oracle_mean_full_universe(summary, oracle_scores)
        if oracle_mean is None:
            oracle_mean = (
                sum(oracle_scores.values()) / len(oracle_scores) if oracle_scores else 0.0
            )
            oracle_note = (
                "subset denominator: full task universe unavailable "
                "(no summary oracle_mean_score, no seed traces) — oracle_mean/"
                "oracle_gap may be inflated if any task lacked a finite score"
            )

        # Contributing chains
        contributing = set(entry["candidate_id"] for entry in ptb.values())

        # Strategy coverage
        coverage: dict[str, dict[str, float]] = {}
        for cat_name in set(list(TASK_CATEGORIES.keys()) + ["other"]):
            cat_tasks = {
                tid: score for tid, score in oracle_scores.items()
                if OmegaEngine.categorize_task(tid) == cat_name
            }
            if not cat_tasks:
                continue
            n = len(cat_tasks)
            coverage[cat_name] = {
                "count": n,
                "above_0.5": round(sum(1 for s in cat_tasks.values() if s > 0.5) / n, 3),
                "above_0.8": round(sum(1 for s in cat_tasks.values() if s > 0.8) / n, 3),
            }

        result = {
            "oracle_mean": round(oracle_mean, 4),
            "best_single_mean": round(best_mean, 4),
            "oracle_gap": round(oracle_mean - best_mean, 4),
            "contributing_chains": len(contributing),
            "total_candidates": index.get("size", 0),
            "contribution_ratio": round(
                len(contributing) / max(index.get("size", 1), 1), 3
            ),
            "strategy_coverage": coverage,
        }
        if oracle_note:
            result["oracle_denominator_note"] = oracle_note

        # Pairwise candidate distance (optional, needs embeddings)
        if self._use_embeddings:
            distances = self._compute_pairwise_distances(index)
            if distances:
                result["pairwise_candidate_distance"] = distances

        return result

    def _oracle_mean_full_universe(
        self, summary: dict, oracle_scores: dict[str, float]
    ) -> float | None:
        """Oracle mean over the run's FULL task universe (missing task → 0.0).

        Prefers summary.json's ``oracle_mean_score`` (both the mid-run and
        final shapes already carry the all-tasks average, so metrics.json
        agrees with the run's own summary); else derives the universe from the
        seed candidate's traces (every task is evaluated at seed). ``None``
        when neither source is available.
        """
        val = summary.get("oracle_mean_score")
        if isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val):
            return float(val)
        seed_task_ids = {
            t.get("task_id") for t in self._get_seed_traces() if t.get("task_id")
        }
        if seed_task_ids:
            universe = seed_task_ids | set(oracle_scores)
            return sum(oracle_scores.get(tid, 0.0) for tid in universe) / len(universe)
        return None

    def _compute_pairwise_distances(self, index: dict) -> dict | None:
        model = self._get_embedding_model()
        if not model:
            return None

        candidates = index.get("candidates", [])
        texts = []
        valid_ids = []
        for c in candidates:
            cand_id = c["candidate_id"]
            codes = self._load_injected_codes_for_candidate(cand_id)
            if not codes:
                continue
            # T3.7: include the code_library channel (helper VALUES) so pairwise
            # candidate distance reflects pure-helper layers, not prompts only.
            combined = "\n".join(
                "\n".join(filter(None, [
                    ic.pre_process, ic.rationale,
                    *ic.code_library.values(), *ic.code_library_bash.values(),
                ]))
                for ic in codes
            )
            if combined.strip():
                texts.append(combined)
                valid_ids.append(cand_id)

        if len(texts) < 2:
            return None

        import numpy as np
        embeddings = model.encode(texts)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        normalized = embeddings / norms
        sim_matrix = normalized @ normalized.T
        n = len(texts)
        distances = []
        for i in range(n):
            for j in range(i + 1, n):
                distances.append(1.0 - float(sim_matrix[i, j]))

        return {
            "mean": round(float(np.mean(distances)), 4),
            "std": round(float(np.std(distances)), 4),
            "min": round(float(np.min(distances)), 4),
            "max": round(float(np.max(distances)), 4),
            "n_candidates": n,
        }

    # ------------------------------------------------------------------
    # Metric 4: Failure Pattern Reduction
    # ------------------------------------------------------------------

    def compute_failure_patterns(self) -> dict:
        """Compare error distributions between seed and best candidate."""
        seed_traces = self._get_seed_traces()
        best_traces = self._get_best_candidate_traces()

        if not seed_traces:
            return {"skipped": "no seed traces"}

        seed_errors = self._classify_trace_errors(seed_traces)
        best_errors = self._classify_trace_errors(best_traces)

        # Error reduction
        all_types = set(list(seed_errors["distribution"].keys()) + list(best_errors["distribution"].keys()))
        reduction = {}
        for t in all_types:
            seed_count = seed_errors["distribution"].get(t, 0)
            best_count = best_errors["distribution"].get(t, 0)
            reduction[t] = seed_count - best_count

        # Failure rate trajectory
        trajectory = []
        if self.mode == "evolutionary":
            index = self._load_archive_index()
            candidates = index.get("candidates", []) if index else []
            # For each iteration, use the best candidate's failure rate
            by_iter = _best_candidate_by_iteration(candidates)
            for i in sorted(by_iter.keys()):
                c = by_iter[i]
                n_total = c.get("pass_at_1", 0)
                # pass_at_1 is fraction, so failure rate = 1 - pass_at_1
                trajectory.append(round(1.0 - n_total, 3))
        else:
            summary = self._load_summary()
            pass_at_1 = summary.get("pass_at_1", {})
            for d in sorted(int(k) for k in pass_at_1):
                trajectory.append(round(1.0 - pass_at_1[str(d)], 3))

        return {
            "seed_errors": seed_errors,
            "best_candidate_errors": best_errors,
            "error_reduction": reduction,
            "failure_rate_trajectory": trajectory,
        }

    @staticmethod
    def _classify_trace_errors(traces: list[dict]) -> dict:
        failures = [t for t in traces if not t.get("success", False)]
        if not failures:
            return {"total_failures": 0, "distribution": {}}

        counts: Counter = Counter()
        for t in failures:
            trace = Trace(
                task_id=t.get("task_id", ""),
                error_summary=t.get("error_summary", ""),
                stderr=t.get("stderr", ""),
                terminated_by=t.get("terminated_by", ""),
            )
            # classify_error is the public taxonomy in meta_layer;
            # OmegaEngine._classify_error is a staticmethod alias of the same
            # function, so this call is byte-identical in output.
            error_type = classify_error(trace)
            counts[error_type] += 1

        return {
            "total_failures": len(failures),
            "distribution": dict(counts.most_common()),
        }

    # ------------------------------------------------------------------
    # Metric 5: Robustness
    # ------------------------------------------------------------------

    def compute_robustness(self) -> dict:
        """Compute regression rates, dev-test gap, and score variance."""
        seed_traces = self._get_seed_traces()
        best_traces = self._get_best_candidate_traces()

        if not seed_traces or not best_traces:
            return {"skipped": "missing traces"}

        seed_scores = {t["task_id"]: t.get("score", 0.0) for t in seed_traces}
        best_scores = {t["task_id"]: t.get("score", 0.0) for t in best_traces}

        # Regression rate
        regressions = []
        for tid in seed_scores:
            if tid in best_scores and best_scores[tid] < seed_scores[tid] - 0.01:
                regressions.append({
                    "task_id": tid,
                    "seed_score": round(seed_scores[tid], 3),
                    "best_score": round(best_scores[tid], 3),
                    "delta": round(best_scores[tid] - seed_scores[tid], 3),
                })

        n_shared = len(set(seed_scores) & set(best_scores))
        regression_rate = len(regressions) / n_shared if n_shared > 0 else 0.0

        result: dict[str, Any] = {
            "regression_rate": round(regression_rate, 3),
            "num_regressions": len(regressions),
            "regressions": sorted(regressions, key=lambda x: x["delta"]),
        }

        # Dev-test gap
        test_results = self._load_test_results()
        if test_results:
            summary = self._load_summary()
            dev_scores = summary.get("per_task_best_scores", {})
            test_scores = test_results.get("test_scores", {})
            gaps = {}
            for tid in dev_scores:
                if tid in test_scores:
                    gaps[tid] = {
                        "dev": round(dev_scores[tid], 3),
                        "test": round(test_scores[tid], 3),
                        "gap": round(dev_scores[tid] - test_scores[tid], 3),
                    }
            mean_gap = (
                sum(g["gap"] for g in gaps.values()) / len(gaps) if gaps else 0.0
            )
            # 6.4a: the SIGN of dev-test is PROPERTY-DEPENDENT, not a universal
            # overfit signal. dev>test (positive) = the dev split overfit; test>dev
            # (negative, e.g. CO-Bench-gpt5.2) = the dev split was pessimistic and
            # the system GENERALIZES — GOOD, never flagged as overfit.
            for g in gaps.values():
                g["direction"] = dev_test_direction(g["gap"])
            result["dev_test_gap"] = {
                "mean_gap": round(mean_gap, 3),
                "direction": dev_test_direction(mean_gap),
                "per_task": gaps,
                "note": "sign is property-dependent: dev>test=overfit; test>dev=generalizes (NOT overfit)",
            }
        else:
            result["dev_test_gap"] = {"skipped": "no test_results.json"}

        # Score variance by depth (evolutionary only)
        if self.mode == "evolutionary":
            index = self._load_archive_index()
            if index:
                candidates = index.get("candidates", [])
                by_depth: dict[int, list[float]] = {}
                for c in candidates:
                    d = c.get("depth", 1)
                    by_depth.setdefault(d, []).append(c.get("mean_score", 0.0))
                variance = {}
                for d, scores in sorted(by_depth.items()):
                    import statistics
                    variance[d] = {
                        "mean": round(statistics.mean(scores), 3),
                        "std": round(statistics.stdev(scores), 3) if len(scores) > 1 else 0.0,
                        "n_candidates": len(scores),
                    }
                result["score_variance_by_depth"] = variance

        return result

    # ------------------------------------------------------------------
    # Per-task progression (for heatmap)
    # ------------------------------------------------------------------

    def compute_per_task_progression(self) -> dict:
        """Build task × step score matrix showing per-task improvement trajectory.

        Linear mode: task × depth matrix.
        Evolutionary mode: task × iteration matrix (best candidate per iteration).
        Also computes score delta from seed for each cell.
        """
        if self.mode == "linear":
            return self._per_task_progression_linear()
        return self._per_task_progression_evolutionary()

    def _per_task_progression_linear(self) -> dict:
        summary = self._load_summary()
        mean_scores = summary.get("mean_scores", {})
        depths = sorted(int(k) for k in mean_scores)

        tasks: list[str] = []
        matrix: dict[str, dict[int, float]] = {}  # task_id -> {depth: score}

        for depth in depths:
            traces = self._load_depth_traces(depth)
            for t in traces:
                tid = t["task_id"]
                if tid not in matrix:
                    matrix[tid] = {}
                    tasks.append(tid)
                matrix[tid][depth] = t.get("score", 0.0)

        tasks.sort()

        # Build the table and delta from seed
        rows = []
        for tid in tasks:
            scores = [matrix[tid].get(d, 0.0) for d in depths]
            seed_score = scores[0] if scores else 0.0
            deltas = [round(s - seed_score, 3) for s in scores]
            rows.append({
                "task_id": tid,
                "scores": [round(s, 3) for s in scores],
                "deltas": deltas,
            })

        return {
            "steps": depths,
            "step_label": "depth",
            "tasks": tasks,
            "rows": rows,
        }

    def _per_task_progression_evolutionary(self) -> dict:
        index = self._load_archive_index()
        if not index:
            return {"skipped": "no archive index"}

        candidates = index.get("candidates", [])

        # Group candidates by iteration, pick best per iteration
        by_iter = _best_candidate_by_iteration(candidates)

        iterations = sorted(by_iter.keys())
        if not iterations:
            return {"skipped": "no candidates"}

        # Collect all task ids
        all_tasks: set[str] = set()
        for it in iterations:
            all_tasks.update(by_iter[it].get("per_task_scores", {}).keys())
        tasks = sorted(all_tasks)

        # Build matrix
        rows = []
        seed_scores = by_iter[iterations[0]].get("per_task_scores", {})
        for tid in tasks:
            scores = []
            for it in iterations:
                s = by_iter[it].get("per_task_scores", {}).get(tid, 0.0)
                scores.append(round(s, 3))
            seed_s = seed_scores.get(tid, 0.0)
            deltas = [round(s - seed_s, 3) for s in scores]
            rows.append({
                "task_id": tid,
                "scores": scores,
                "deltas": deltas,
            })

        return {
            "steps": iterations,
            "step_label": "iteration",
            "tasks": tasks,
            "rows": rows,
        }

    # ------------------------------------------------------------------
    # Top-level
    # ------------------------------------------------------------------

    def run_all(self) -> dict:
        """Compute all metrics."""
        config = self._load_config()

        return {
            "experiment_dir": str(self.experiment_dir),
            "mode": self.mode,
            "config": config,
            "convergence": self.compute_convergence(),
            "emergent_abstraction": self.compute_emergent_abstraction(),
            "archive_diversity": self.compute_archive_diversity(),
            "failure_patterns": self.compute_failure_patterns(),
            "robustness": self.compute_robustness(),
            "per_task_progression": self.compute_per_task_progression(),
        }

    def save_results(self, results: dict, output_dir: str | None = None) -> Path:
        """Save metrics to experiment directory."""
        out = Path(output_dir) if output_dir else self.experiment_dir / "analysis"
        out.mkdir(parents=True, exist_ok=True)

        with open(out / "metrics.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        return out

    def plot_all(self, results: dict, output_dir: str | None = None) -> None:
        """Generate visualization plots (requires matplotlib)."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.info("matplotlib not installed, skipping plots")
            return

        out = Path(output_dir) if output_dir else self.experiment_dir / "analysis"
        out.mkdir(parents=True, exist_ok=True)

        self._plot_convergence(results.get("convergence", {}), out, plt)
        self._plot_failure_distribution(results.get("failure_patterns", {}), out, plt)
        self._plot_abstraction(results.get("emergent_abstraction", {}), out, plt)
        self._plot_task_heatmap(results.get("per_task_progression", {}), out, plt)

        logger.info("Plots saved to %s", out)

    @staticmethod
    def _plot_convergence(data: dict, out: Path, plt) -> None:
        history = data.get("convergence_history", [])
        if not history:
            return

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(range(len(history)), history, "b-o", label="Best Score", linewidth=2)
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Best Mean Score", color="b")
        ax1.set_ylim(0, 1)

        cumulative = data.get("cumulative_tokens", [])
        if cumulative:
            ax2 = ax1.twinx()
            ax2.bar(range(len(cumulative)), cumulative, alpha=0.2, color="gray", label="Cumulative Tokens")
            ax2.set_ylabel("Cumulative Tokens", color="gray")

        ax1.set_title("Convergence Curve")
        ax1.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(out / "convergence_curve.png", dpi=150)
        plt.close(fig)

    @staticmethod
    def _plot_failure_distribution(data: dict, out: Path, plt) -> None:
        seed = data.get("seed_errors", {}).get("distribution", {})
        best = data.get("best_candidate_errors", {}).get("distribution", {})
        if not seed and not best:
            return

        all_types = sorted(set(list(seed.keys()) + list(best.keys())))
        if not all_types:
            return

        x = range(len(all_types))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar([i - width / 2 for i in x], [seed.get(t, 0) for t in all_types],
               width, label="Seed", color="salmon")
        ax.bar([i + width / 2 for i in x], [best.get(t, 0) for t in all_types],
               width, label="Best Candidate", color="steelblue")
        ax.set_xticks(list(x))
        ax.set_xticklabels(all_types, rotation=30, ha="right")
        ax.set_ylabel("Number of Tasks")
        ax.set_title("Failure Pattern Distribution: Seed vs Best")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "failure_distribution.png", dpi=150)
        plt.close(fig)

    @staticmethod
    def _plot_abstraction(data: dict, out: Path, plt) -> None:
        chain = data.get("best_chain", {})
        gradient = chain.get("abstraction_gradient", [])
        if not gradient:
            return

        depths = list(range(2, 2 + len(gradient)))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(depths, gradient, "g-o", linewidth=2, markersize=8)
        ax.set_xlabel("Depth")
        ax.set_ylabel("Abstraction Score")
        ax.set_title("Abstraction Gradient Across Depths")
        ax.set_ylim(0, 1)
        ax.set_xticks(depths)

        # Annotate with top types
        profiles = chain.get("improvement_type_profiles", [])
        for p in profiles:
            d = p["depth"]
            if d - 2 < len(gradient):
                top = ", ".join(p.get("top_types", [])[:2])
                if top:
                    ax.annotate(top, (d, gradient[d - 2]), textcoords="offset points",
                                xytext=(0, 10), ha="center", fontsize=8)

        fig.tight_layout()
        fig.savefig(out / "abstraction_gradient.png", dpi=150)
        plt.close(fig)

    @staticmethod
    def _plot_task_heatmap(data: dict, out: Path, plt) -> None:
        """Plot task × step heatmap showing score deltas from seed."""
        if "skipped" in data or not data.get("rows"):
            return

        try:
            import numpy as np
        except ImportError:
            return

        tasks = data["tasks"]
        steps = data["steps"]
        step_label = data.get("step_label", "step")
        rows = data["rows"]

        if not tasks or not steps:
            return

        # Build score matrix (absolute scores)
        score_matrix = np.zeros((len(tasks), len(steps)))
        for i, row in enumerate(rows):
            for j, s in enumerate(row["scores"]):
                score_matrix[i, j] = s

        # Build delta matrix (change from seed)
        delta_matrix = np.zeros((len(tasks), len(steps)))
        for i, row in enumerate(rows):
            for j, d in enumerate(row["deltas"]):
                delta_matrix[i, j] = d

        # --- Plot 1: Absolute scores heatmap ---
        fig_h = max(6, len(tasks) * 0.3 + 2)
        fig_w = max(6, len(steps) * 1.5 + 3)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        im = ax.imshow(score_matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(len(steps)))
        ax.set_xticklabels([str(s) for s in steps])
        ax.set_xlabel(step_label.capitalize())
        ax.set_yticks(range(len(tasks)))
        ax.set_yticklabels(tasks, fontsize=7)
        ax.set_title("Per-Task Score Progression")

        # Annotate cells with scores
        for i in range(len(tasks)):
            for j in range(len(steps)):
                val = score_matrix[i, j]
                color = "white" if val > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color=color)

        fig.colorbar(im, ax=ax, label="Score", shrink=0.8)
        fig.tight_layout()
        fig.savefig(out / "task_score_heatmap.png", dpi=150)
        plt.close(fig)

        # --- Plot 2: Delta from seed heatmap ---
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        max_abs = max(abs(delta_matrix.min()), abs(delta_matrix.max()), 0.1)
        im = ax.imshow(delta_matrix, aspect="auto", cmap="RdYlGn",
                        vmin=-max_abs, vmax=max_abs)
        ax.set_xticks(range(len(steps)))
        ax.set_xticklabels([str(s) for s in steps])
        ax.set_xlabel(step_label.capitalize())
        ax.set_yticks(range(len(tasks)))
        ax.set_yticklabels(tasks, fontsize=7)
        ax.set_title("Per-Task Score Change from Seed (green=improved, red=regressed)")

        for i in range(len(tasks)):
            for j in range(len(steps)):
                val = delta_matrix[i, j]
                if abs(val) > 0.005:
                    color = "black" if abs(val) < max_abs * 0.5 else "white"
                    ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                            fontsize=6, color=color)

        fig.colorbar(im, ax=ax, label="Δ Score", shrink=0.8)
        fig.tight_layout()
        fig.savefig(out / "task_delta_heatmap.png", dpi=150)
        plt.close(fig)


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Meta^n Experiment Metrics Analysis")
    parser.add_argument("experiment_dir", help="Path to experiment directory")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: <experiment>/analysis/)")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--no-embeddings", action="store_true", help="Skip embedding-dependent metrics")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    analyzer = ExperimentAnalyzer(
        args.experiment_dir,
        use_embeddings=not args.no_embeddings,
    )
    print(f"Experiment: {args.experiment_dir} (mode: {analyzer.mode})")

    results = analyzer.run_all()
    out = analyzer.save_results(results, args.output_dir)
    print(f"Metrics saved to {out / 'metrics.json'}")

    if not args.no_plots:
        analyzer.plot_all(results, str(out))
        print(f"Plots saved to {out}/")

    # Print summary
    conv = results.get("convergence", {})
    print(f"\nConvergence: {conv.get('final_score', 0):.3f} in {conv.get('total_steps', 0)} steps")

    div = results.get("archive_diversity", {})
    if "oracle_gap" in div:
        print(f"Oracle gap: {div['oracle_gap']:.3f} ({div['contributing_chains']} chains)")

    fp = results.get("failure_patterns", {})
    seed_f = fp.get("seed_errors", {}).get("total_failures", 0)
    best_f = fp.get("best_candidate_errors", {}).get("total_failures", 0)
    print(f"Failures: {seed_f} (seed) → {best_f} (best)")

    rob = results.get("robustness", {})
    print(f"Regressions: {rob.get('num_regressions', 0)} tasks")


if __name__ == "__main__":
    main()
