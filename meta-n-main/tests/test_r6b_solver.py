"""§6b F075 — balanced-JSON fallback for classification extraction (default OFF).

``Layer1Solver._extract_json`` ordering contract:

    fenced -> flat regex -> (``balanced_json_fallback`` ON only) balanced
    scan -> raw response

The flag can therefore ONLY change outcomes where the flag-less code returns
the raw prose. OFF (the default, including ``__new__``-constructed instances)
is byte-identical to the pre-F075 algorithm; ON recovers string-aware
brace-balanced objects (nested values, braces inside label strings) that the
flat ``[^{}]*`` regex cannot match, validated via ``json.loads`` + a
``case_<n>`` key check.
"""

import re
from unittest.mock import MagicMock

from meta_n.core.solver import Layer1Solver, _balanced_json_object, extract_fenced_block

# The pre-F075 extraction algorithm, inlined as the byte-identity reference.
def _legacy_extract_json(response: str) -> str:
    result = extract_fenced_block(response, ("json", ""))
    if result is not None:
        return result
    match = re.search(r"\{[^{}]*(?:\"case_\d+\"[^{}]*)+\}", response, re.DOTALL)
    if match:
        return match.group(0)
    return response.strip()


# Responses the flat regex CANNOT extract (legacy returns raw prose).
NESTED_UNFENCED = 'prose {"case_1": {"label": "flu"}, "case_2": "cold"} prose'
BRACE_IN_VALUE = 'answer: {"case_1": "a{b"} thanks'
STRAY_GROUP_THEN_NESTED = 'note {x} then {"case_1": {"label": "flu"}}'
ESCAPED_QUOTE_IN_VALUE = 'so: {"case_1": "say \\"hi\\" {ok}"} end'
SINGLE_QUOTED = "{'case_1': 'flu'}"
BRACE_FREE_PROSE = "no json here at all"

# Corpus shared with tests/test_refine_solvers_llm.py (flat / fenced cases the
# flag must not disturb) plus the recoverable and non-recoverable shapes above.
CORPUS = [
    "```bash\necho hi\n```",
    "```sh\necho sh\n```",
    "```\nplain\n```",
    "no fence at all",
    "```python\nx = 1\n```",
    "```py\ny = 2\n```",
    "```json\n{\"case_1\": \"a\"}\n```",
    "prefix {\"case_1\": \"a\", \"case_2\": \"b\"} suffix",
    "```bash\n```",
    NESTED_UNFENCED,
    BRACE_IN_VALUE,
    STRAY_GROUP_THEN_NESTED,
    ESCAPED_QUOTE_IN_VALUE,
    SINGLE_QUOTED,
    BRACE_FREE_PROSE,
    # Two case_ keys around a brace-in-value: the flat regex matches a garbage
    # span starting INSIDE the string ('{b", "case_2": "c"}') — regex wins
    # before the balanced scan, so ON and OFF must both return that span.
    '{"case_1": "a{b", "case_2": "c"}',
]


class TestOffContractByteIdentity:
    def test_default_off_matches_legacy_algorithm_on_corpus(self):
        solver = Layer1Solver(MagicMock())
        assert solver.balanced_json_fallback is False
        for response in CORPUS:
            assert solver._extract_json(response) == _legacy_extract_json(response)

    def test_off_returns_whole_prose_for_recoverable_shapes(self):
        """Pins the exact defect shape OFF preserves: nested / brace-in-value
        objects fall through to the raw response."""
        solver = Layer1Solver(MagicMock())
        assert solver._extract_json(NESTED_UNFENCED) == NESTED_UNFENCED.strip()
        assert solver._extract_json(BRACE_IN_VALUE) == BRACE_IN_VALUE.strip()

    def test_off_constructor_kwarg_is_explicitly_false_by_default(self):
        solver = Layer1Solver(MagicMock())
        assert solver.balanced_json_fallback is False
        assert Layer1Solver.balanced_json_fallback is False

    def test_new_constructed_instance_defaults_off(self):
        solver = Layer1Solver.__new__(Layer1Solver)
        assert solver.balanced_json_fallback is False


class TestOnBehavior:
    @staticmethod
    def _solver_on() -> Layer1Solver:
        return Layer1Solver(MagicMock(), balanced_json_fallback=True)

    def test_nested_object_recovered(self):
        result = self._solver_on()._extract_json(NESTED_UNFENCED)
        assert result == '{"case_1": {"label": "flu"}, "case_2": "cold"}'

    def test_brace_in_value_recovered(self):
        """Flat, contract-compliant map whose label value contains a brace —
        the flat regex cannot match it (single case key), the balanced scan can."""
        result = self._solver_on()._extract_json(BRACE_IN_VALUE)
        assert result == '{"case_1": "a{b"}'

    def test_stray_group_skipped_to_valid_object(self):
        result = self._solver_on()._extract_json(STRAY_GROUP_THEN_NESTED)
        assert result == '{"case_1": {"label": "flu"}}'

    def test_escaped_quote_inside_value_recovered(self):
        result = self._solver_on()._extract_json(ESCAPED_QUOTE_IN_VALUE)
        assert result == '{"case_1": "say \\"hi\\" {ok}"}'

    def test_single_quoted_pseudo_json_still_raw(self):
        """json.loads validation rejects non-JSON; ON still returns raw prose."""
        assert self._solver_on()._extract_json(SINGLE_QUOTED) == SINGLE_QUOTED

    def test_brace_free_prose_still_raw(self):
        assert self._solver_on()._extract_json(BRACE_FREE_PROSE) == BRACE_FREE_PROSE

    def test_no_divergence_where_fenced_or_regex_wins(self):
        """For every corpus response the flag-less path already extracts
        (fenced or flat regex), ON returns the identical string — the flag
        only extends the raw-prose tail."""
        off = Layer1Solver(MagicMock())
        on = self._solver_on()
        for response in CORPUS:
            if _legacy_extract_json(response) != response.strip():
                assert on._extract_json(response) == off._extract_json(response)


class TestBalancedJsonObjectHelper:
    def test_returns_none_without_case_key(self):
        assert _balanced_json_object('{"label": "flu"}') is None

    def test_returns_none_for_non_dict_json(self):
        assert _balanced_json_object('["case_1"]') is None

    def test_returns_none_for_unbalanced_braces(self):
        assert _balanced_json_object('{"case_1": "flu"') is None

    def test_braces_inside_strings_do_not_count_depth(self):
        span = _balanced_json_object('x {"case_1": "}{"} y')
        assert span == '{"case_1": "}{"}'

    def test_first_valid_object_wins(self):
        text = '{"case_1": "a"} and {"case_2": "b"}'
        assert _balanced_json_object(text) == '{"case_1": "a"}'
