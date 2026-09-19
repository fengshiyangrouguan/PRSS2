"""Emergent role analysis for Meta^n layers (Experiment 6).

Post-hoc analysis module that reads saved experiment artifacts and produces:
- Improvement type classification per layer
- Abstraction level scoring
- Inter-layer diversity via embedding similarity
- Role differentiation scores

Heuristic-divergence contract: the abstraction / improvement-type heuristics in
this module are the Exp-6 variants and are DELIBERATELY different algorithms
from ``meta_n.analysis.metrics.ExperimentAnalyzer``'s same-purpose heuristics
(different keyword sets, per-task literal-phrase matching, AST signals, 7-type
vs 5-type score dicts, different normalization). Their numeric outputs are NOT
comparable across the two files — the persisted ``role_analysis.json`` keys
carry an ``_exp6`` suffix (schema v2, ``ROLE_ANALYSIS_SCHEMA_VERSION``) to mark
that incomparability against ``metrics.json``'s same-purpose fields. Do not
fold the two implementations together (which algorithm wins is a research
decision); channel-fold changes of the T3.6/T3.7 class must be applied to BOTH
copies in parallel.

Usage:
    python -m meta_n.analysis.emergent_roles --experiment-dir ./experiments/run_001/
"""

from __future__ import annotations

import ast
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional
import numpy as np

from meta_n.analysis.metrics import _load_contiguous_injected_codes
from meta_n.core.meta_layer import InjectedCode, TaskDescription

logger = logging.getLogger(__name__)


class ImprovementType(str, Enum):
    """Classification of what kind of improvement a layer made."""

    UTILITY = "utility"
    PROMPT_MOD = "prompt_mod"
    CONTROL_FLOW = "control_flow"
    RETRY = "retry"
    DECOMPOSITION = "decomposition"
    ERROR_HANDLING = "error_handling"
    STRATEGY_SELECTION = "strategy_selection"


@dataclass
class LayerRoleProfile:
    """Complete role analysis for a single layer."""

    depth: int
    improvement_types: list[ImprovementType] = field(default_factory=list)
    improvement_type_scores: dict[str, float] = field(default_factory=dict)
    # None = ungrounded (no task descriptions available), serialized as null.
    abstraction_score: Optional[float] = 0.5
    rationale_embedding: Optional[np.ndarray] = None
    code_embedding: Optional[np.ndarray] = None
    combined_embedding: Optional[np.ndarray] = None
    referenced_depths: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        # Serialized-key mapping (role_analysis.json schema v2): the numeric
        # heuristic outputs carry an ``_exp6`` suffix so they cannot be read as
        # comparable with metrics.json's same-purpose fields (different
        # algorithms — see the module docstring). Python attribute names stay
        # unsuffixed; the collision was in the persisted JSON, not the API.
        #   abstraction_score        -> "abstraction_score_exp6"
        #   improvement_type_scores  -> "improvement_type_scores_exp6"
        #   improvement_types        -> "improvement_types" (a label list, not a
        #       numeric metric; metrics.json's counterpart is already the
        #       differently-named "top_types")
        return {
            "depth": self.depth,
            "improvement_types": [t.value for t in self.improvement_types],
            "improvement_type_scores_exp6": self.improvement_type_scores,
            "abstraction_score_exp6": self.abstraction_score,
            "referenced_depths": self.referenced_depths,
            "has_embeddings": self.combined_embedding is not None,
        }


#: ``role_analysis.json`` schema version. v1 = version-key-absent files with the
#: legacy unsuffixed key names (still valid v1 documents on disk; the artifact is
#: post-hoc and regenerable by re-running this CLI). v2 = the ``_exp6``-suffixed
#: numeric-metric keys marking incomparability with metrics.json (module
#: docstring). Bump whenever the serialized field set or key names change.
ROLE_ANALYSIS_SCHEMA_VERSION: int = 2


@dataclass
class RoleAnalysisResult:
    """Complete Exp 6 analysis output."""

    experiment_dir: str = ""
    layer_profiles: list[LayerRoleProfile] = field(default_factory=list)
    pairwise_distances: dict[str, float] = field(default_factory=dict)
    mean_differentiation_score: float = 0.0
    abstraction_gradient: list[Optional[float]] = field(default_factory=list)
    injection_targeting: dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        # Schema v2: version key first; ``abstraction_gradient`` serializes as
        # "abstraction_gradient_exp6" (Exp-6 heuristic — NOT comparable with
        # metrics.json's best_chain.abstraction_gradient; see module docstring).
        return {
            "role_analysis_schema_version": ROLE_ANALYSIS_SCHEMA_VERSION,
            "experiment_dir": self.experiment_dir,
            "mean_differentiation_score": self.mean_differentiation_score,
            "abstraction_gradient_exp6": self.abstraction_gradient,
            "injection_targeting": {str(k): v for k, v in self.injection_targeting.items()},
            "pairwise_distances": self.pairwise_distances,
            "layer_profiles": [p.to_dict() for p in self.layer_profiles],
        }


# --- Heuristic keyword sets for classification ---

_TYPE_KEYWORDS: dict[str, list[str]] = {
    "utility": ["utility", "helper", "reusable", "function", "def "],
    "prompt_mod": ["hint", "context", "prompt", "instruction", "additional_context", "guidance"],
    "control_flow": ["conditional", "branch", "if ", "elif ", "switch", "match "],
    "retry": ["retry", "fallback", "attempt", "again", "re-run", "loop"],
    "decomposition": ["decompose", "break down", "sub-task", "step", "split", "phase"],
    "error_handling": ["error", "handle", "catch", "recover", "exception", "try", "except"],
    "strategy_selection": ["classify", "type", "category", "strategy", "select", "choose", "detect"],
}


class EmergentRoleAnalyzer:
    """Post-hoc analysis of emergent roles in Meta^n layer stacks."""

    def __init__(
        self,
        experiment_dir: str,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.experiment_dir = Path(experiment_dir)
        self.embedding_model_name = embedding_model
        self._embedding_model = None  # lazy load

    def analyze(self, tasks: list[TaskDescription] | None = None) -> RoleAnalysisResult:
        """Run the full analysis pipeline."""
        injected_codes = self._load_injected_codes()
        if not injected_codes:
            return RoleAnalysisResult(experiment_dir=str(self.experiment_dir))

        if tasks is None:
            tasks = self._load_tasks()
        if not tasks:
            logger.warning(
                "No task descriptions available (config.json 'tasks_file' is "
                "unset — the default for --benchmark runs): the Exp-6 "
                "task-specific term cannot be computed, so "
                "abstraction_score_exp6 / abstraction_gradient_exp6 will be "
                "null instead of a degenerate 'entirely generic' score."
            )

        # 1. Build layer profiles with heuristic classification
        profiles = []
        for code in injected_codes:
            profile = LayerRoleProfile(depth=code.source_depth)
            profile.improvement_type_scores = self._classify_improvement_type(code)
            profile.improvement_types = self._top_types(profile.improvement_type_scores)
            profile.abstraction_score = self._compute_abstraction_score(code, tasks)
            profile.referenced_depths = self._extract_referenced_depths(code)
            profiles.append(profile)

        # 2. Compute embeddings
        self._compute_embeddings(profiles, injected_codes)

        # 3. Pairwise role differentiation
        pairwise = self._compute_role_differentiation(profiles)
        dists = list(pairwise.values())
        mean_diff = float(np.mean(dists)) if dists else 0.0

        # 4. Injection targeting accuracy
        targeting = self._compute_injection_targeting(profiles)

        return RoleAnalysisResult(
            experiment_dir=str(self.experiment_dir),
            layer_profiles=profiles,
            pairwise_distances=pairwise,
            mean_differentiation_score=mean_diff,
            abstraction_gradient=[p.abstraction_score for p in profiles],
            injection_targeting=targeting,
        )

    def save_results(self, result: RoleAnalysisResult, output_dir: str | None = None):
        """Save analysis results to JSON + embeddings as .npy."""
        out = Path(output_dir or str(self.experiment_dir / "analysis"))
        out.mkdir(parents=True, exist_ok=True)

        with open(out / "role_analysis.json", "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        # Save embeddings as numpy arrays
        for profile in result.layer_profiles:
            if profile.combined_embedding is not None:
                np.save(
                    out / f"embedding_depth_{profile.depth}.npy",
                    profile.combined_embedding,
                )

    # --- Classification ---

    def _classify_improvement_type(self, code: InjectedCode) -> dict[str, float]:
        """Classify improvement type via heuristic AST + keyword analysis.

        Exp-6 heuristic — DELIBERATELY different from
        ``metrics.ExperimentAnalyzer._classify_improvement_types`` (7-type
        ``ImprovementType`` enum, 0.2/hit capped at 0.6, AST signals via
        ``_ast_signals``, max-normalization vs metrics' 5 types and
        ``min(count/(len*0.5), 1)``). Scores are numerically incomparable across
        the two files; the serialized key carries the ``_exp6`` suffix to say
        so. Do not consolidate; T3.6/T3.7-class channel-fold changes must be
        applied to both copies in parallel.
        """
        scores: dict[str, float] = {t.value: 0.0 for t in ImprovementType}

        # T3.6: fold the code_library channel (Python + bash helper VALUES) into
        # the analyzed text so a pure-helper layer is not mis-scored utility=0.
        all_text = "\n".join(filter(None, [
            code.pre_process, code.rationale,
            *code.code_library.values(), *code.code_library_bash.values(),
        ])).lower()

        # Keyword scoring
        for type_name, keywords in _TYPE_KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw.lower() in all_text)
            scores[type_name] += min(hits * 0.2, 0.6)

        # AST-based signals (pre_process + Python code_library values; bash helpers
        # are not valid Python and are intentionally excluded from the AST input).
        py_source = "\n".join(filter(None, [
            code.pre_process, *code.code_library.values(),
        ]))
        if py_source:
            try:
                tree = ast.parse(py_source)
                self._ast_signals(tree, scores)
            except SyntaxError:
                pass

        if code.pre_process and "additional_context" in code.pre_process:
            scores["prompt_mod"] += 0.4

        # Normalize to [0, 1]
        max_score = max(scores.values()) if scores else 1.0
        if max_score > 0:
            scores = {k: min(v / max_score, 1.0) for k, v in scores.items()}

        return scores

    def _ast_signals(self, tree: ast.AST, scores: dict[str, float]):
        """Extract classification signals from AST."""
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.Match)):
                scores["control_flow"] += 0.2
                scores["strategy_selection"] += 0.1

            elif isinstance(node, ast.Try):
                scores["error_handling"] += 0.3

            elif isinstance(node, (ast.For, ast.While)):
                scores["retry"] += 0.15
                scores["control_flow"] += 0.1

            elif isinstance(node, ast.FunctionDef):
                scores["utility"] += 0.15
                scores["decomposition"] += 0.1

    def _top_types(
        self, scores: dict[str, float], threshold: float = 0.3
    ) -> list[ImprovementType]:
        """Return improvement types above threshold, sorted by score."""
        return [
            ImprovementType(k)
            for k, v in sorted(scores.items(), key=lambda x: -x[1])
            if v >= threshold
        ]

    # --- Abstraction scoring ---

    def _compute_abstraction_score(
        self, code: InjectedCode, tasks: list[TaskDescription]
    ) -> Optional[float]:
        """
        Measure ratio of task-specific vs generic patterns.
        0.0 = entirely task-specific, 1.0 = entirely generic.

        Returns ``None`` when ``tasks`` is empty: without task text the
        task-specific term is structurally zero and every layer would score a
        fake binary {0.5, 1.0} — even one that hardcodes a task_id.

        Exp-6 heuristic — DELIBERATELY different from
        ``metrics.ExperimentAnalyzer._compute_abstraction_score`` (different
        keyword lists, plus per-task 4-gram literal-phrase matching against
        ``tasks``, which metrics' copy does not do). Scores are numerically
        incomparable across the two files; the serialized keys
        (``abstraction_score_exp6`` / ``abstraction_gradient_exp6``) carry the
        suffix to say so. Do not consolidate; T3.6/T3.7-class channel-fold
        changes must be applied to both copies in parallel.
        """
        if not tasks:
            return None
        # T3.6: include the code_library channel (helper VALUES) so a pure-helper
        # layer's abstraction is measured over its actual code, not just prompts.
        all_code = "\n".join(filter(None, [
            code.pre_process, code.rationale,
            *code.code_library.values(), *code.code_library_bash.values(),
        ]))
        if not all_code:
            return 0.5

        all_code_lower = all_code.lower()
        task_specific_refs = 0

        for task in tasks:
            task_specific_refs += all_code.count(task.task_id)
            # Check for literal phrases from task descriptions
            words = task.description.split()
            for i in range(len(words) - 3):
                phrase = " ".join(words[i:i + 4]).lower()
                if phrase in all_code_lower:
                    task_specific_refs += 1

        generic_keywords = [
            "task.description", "task.task_id", ".lower()", "for ", "if ",
            "def ", "lambda", "pattern", "classify", "any(", "all(",
            "isinstance", "type(", "in task",
        ]
        generic_refs = sum(1 for kw in generic_keywords if kw in all_code)

        total = task_specific_refs + generic_refs
        if total == 0:
            return 0.5

        return generic_refs / total

    # --- Embeddings ---

    def _get_embedding_model(self):
        """Lazy-load sentence-transformers model.

        Raises on ImportError (the caller, ``_compute_embeddings``, catches and
        skips embeddings). The ``metrics.ExperimentAnalyzer._get_embedding_model``
        twin has a deliberately different error contract (degrades to ``None``
        in-place) — do not consolidate.
        """
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model

    def _compute_embeddings(
        self, profiles: list[LayerRoleProfile], codes: list[InjectedCode]
    ):
        """Compute embeddings for rationales and code. Mutates profiles in-place."""
        try:
            model = self._get_embedding_model()
        except Exception:
            # sentence-transformers not installed or model unavailable
            return

        for profile, code in zip(profiles, codes):
            # Rationale embedding
            rationale_text = code.rationale or ""
            if rationale_text:
                profile.rationale_embedding = model.encode(rationale_text)

            # Code embedding — T3.6: embed the code_library channel (helper
            # VALUES) alongside pre_process so a pure-helper layer is not blank.
            code_text = "\n".join(filter(None, [
                code.pre_process,
                *code.code_library.values(), *code.code_library_bash.values(),
            ]))
            if code_text:
                profile.code_embedding = model.encode(code_text)

            # Combined — always a FIXED 2*D vector regardless of which channels
            # are present, so adjacent layers with differing channel presence
            # stay shape-compatible for the np.dot() in
            # _compute_role_differentiation. A missing channel is zero-padded
            # (its half contributes nothing) rather than collapsing the vector
            # to a bare D-dim array (which would raise ValueError on np.dot of
            # unequal-length arrays).
            rat = profile.rationale_embedding
            cod = profile.code_embedding
            if rat is not None or cod is not None:
                ref = rat if rat is not None else cod
                zeros = np.zeros_like(ref)
                rat_half = (rat if rat is not None else zeros) * 0.5
                cod_half = (cod if cod is not None else zeros) * 0.5
                profile.combined_embedding = np.concatenate([rat_half, cod_half])

    def _compute_role_differentiation(
        self, profiles: list[LayerRoleProfile]
    ) -> dict[str, float]:
        """Compute cosine distance between adjacent layers' embeddings."""
        distances: dict[str, float] = {}

        for i in range(len(profiles) - 1):
            p1, p2 = profiles[i], profiles[i + 1]
            emb1 = p1.combined_embedding
            emb2 = p2.combined_embedding

            if emb1 is not None and emb2 is not None:
                # Cosine distance = 1 - cosine_similarity
                sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)
                dist = 1.0 - float(sim)
                distances[f"{p1.depth}-{p2.depth}"] = dist

        return distances

    # --- Depth references ---

    def _extract_referenced_depths(self, code: InjectedCode) -> list[int]:
        """Extract depth references from rationale text."""
        referenced_depths: list[int] = []
        if code.rationale:
            for match in re.finditer(r"(?:depth|layer)\s*(\d+)", code.rationale, re.IGNORECASE):
                referenced_depths.append(int(match.group(1)))
        return sorted(set(referenced_depths))

    # --- Injection targeting ---

    def _compute_injection_targeting(
        self, profiles: list[LayerRoleProfile]
    ) -> dict[int, float]:
        """
        Score whether each layer targets the appropriate abstraction level.

        Expectation: lower depths → concrete fixes (error_handling, retry)
                     higher depths → abstract strategies (strategy_selection, decomposition)
        """
        concrete_types = {"error_handling", "retry", "control_flow", "utility"}
        abstract_types = {"prompt_mod", "decomposition", "strategy_selection"}

        accuracy: dict[int, float] = {}
        if not profiles:
            return accuracy

        max_depth = max(p.depth for p in profiles)

        for profile in profiles:
            if profile.depth < 2 or not profile.improvement_types:
                continue

            actual = {t.value for t in profile.improvement_types}
            # Normalized position in stack: 0 = bottom, 1 = top
            position = (profile.depth - 2) / max(max_depth - 2, 1)

            if position < 0.5:
                expected = concrete_types
            else:
                expected = abstract_types

            overlap = len(actual & expected) / max(len(actual), 1)
            accuracy[profile.depth] = overlap

        return accuracy

    # --- I/O ---

    def _load_injected_codes(self) -> list[InjectedCode]:
        """Load InjectedCode objects from the experiment directory.

        Tries the linear ``depth_N/injected_code.json`` layout first (merging
        ``depth_N/omega_response.txt`` into ``raw_omega_response`` when
        present). When that yields nothing and an evolutionary archive exists,
        falls back to the best chain's ``archive/<id>/injected_code_d{N}.json``
        files, so evolutionary runs no longer produce an empty analysis.
        """

        def _merge_omega_response(depth: int, data: dict) -> None:
            response_path = (
                self.experiment_dir / f"depth_{depth}" / "omega_response.txt"
            )
            if response_path.exists():
                data["raw_omega_response"] = response_path.read_text()

        codes = _load_contiguous_injected_codes(
            lambda d: self.experiment_dir / f"depth_{d}" / "injected_code.json",
            augment=_merge_omega_response,
        )
        if codes:
            return codes

        if not (self.experiment_dir / "archive" / "index.json").exists():
            return codes

        best_id = self._best_candidate_id()
        if not best_id:
            return []
        # The evolutionary layout persists no per-depth omega-response file
        # (raw_omega_response is excluded from the archived JSON), so there is
        # nothing to merge on this path.
        cand_dir = self.experiment_dir / "archive" / best_id
        return _load_contiguous_injected_codes(
            lambda d: cand_dir / f"injected_code_d{d}.json"
        )

    def _best_candidate_id(self) -> str:
        """Best-chain candidate id: ``lineage/best_chain.json`` first, then the
        run-level ``summary.json`` (mirrors the metrics.py resolution order)."""
        lineage_path = self.experiment_dir / "lineage" / "best_chain.json"
        if lineage_path.exists():
            with open(lineage_path) as f:
                best_id = json.load(f).get("best_candidate_id", "")
            if best_id:
                return best_id
        summary_path = self.experiment_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                return json.load(f).get("best_candidate_id", "") or ""
        return ""

    def _load_tasks(self) -> list[TaskDescription]:
        """Load task descriptions from config.json → tasks file."""
        config_path = self.experiment_dir / "config.json"
        if not config_path.exists():
            return []
        with open(config_path) as f:
            config = json.load(f)

        tasks_file = config.get("tasks_file", "")
        if not tasks_file or not Path(tasks_file).exists():
            return []

        with open(tasks_file) as f:
            return [TaskDescription(**t) for t in json.load(f)]


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Meta^n Emergent Role Analysis (Experiment 6)")
    parser.add_argument("--experiment-dir", required=True, help="Path to experiment directory")
    parser.add_argument(
        "--output-dir", default=None, help="Output directory (default: <experiment>/analysis/)"
    )
    args = parser.parse_args()

    analyzer = EmergentRoleAnalyzer(args.experiment_dir)
    result = analyzer.analyze()
    out = args.output_dir or str(Path(args.experiment_dir) / "analysis")
    analyzer.save_results(result, out)
    print(f"Role analysis saved to {Path(out) / 'role_analysis.json'}")
    print(f"Mean differentiation score: {result.mean_differentiation_score:.3f}")


if __name__ == "__main__":
    main()
