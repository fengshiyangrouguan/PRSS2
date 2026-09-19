"""F127 regression — no val-split payload in text-classification task metadata.

``TaskDescription.metadata`` is readable at runtime by Ω-injected
``pre_process``, whose ``additional_context`` output is appended to the solver
prompt. A val-split payload in metadata (per-case texts or gold labels) would
therefore let injected code leak gold DEV labels into the solver prompt while
dev scores drive archive selection. Evaluation never needs these keys:
``_run_solve_code`` re-derives cases/labels/few_shot from ``_get_data()``.
"""

import json

from meta_n.integrations.text_classification import TextClassificationAdapter

_VAL_TEXT = "fever and chills"


def _stub_adapter() -> TextClassificationAdapter:
    """Offline adapter with stubbed ``_data`` (no HF datasets, no LLM)."""
    adapter = TextClassificationAdapter.__new__(TextClassificationAdapter)
    adapter.dataset_name = "symptom2disease"
    adapter.data_dir = None
    adapter.n_few_shot = 2
    adapter.max_val = 50
    adapter.max_test = None
    adapter._llm_client = None
    adapter._eval_workers = 1
    adapter._data = {
        "train": [
            {"text": "itching and rash", "label": "Fungal infection"},
            {"text": "sneezing and watery eyes", "label": "Allergy"},
            {"text": "headache", "label": "Migraine"},
        ],
        "val": [
            {"text": _VAL_TEXT, "label": "Malaria"},
        ],
        "test": [
            {"text": "joint pain", "label": "Arthritis"},
        ],
        "labels": ["Allergy", "Arthritis", "Fungal infection", "Malaria", "Migraine"],
        "metric": "accuracy",
        "language": "en",
    }
    return adapter


def test_metadata_has_no_val_split_payload():
    tasks = _stub_adapter().load_tasks()
    assert len(tasks) == 1
    metadata = tasks[0].metadata
    assert "val_cases" not in metadata
    assert "val_labels" not in metadata


def test_metadata_carries_no_per_case_gold_mapping():
    # String-level invariant: no val-split payload survives under ANY key
    # (guards against the keys merely being renamed). The label vocabulary
    # (label_set) is allowed; per-case texts/label mappings are not.
    task = _stub_adapter().load_tasks()[0]
    assert _VAL_TEXT not in json.dumps(task.metadata)


def test_description_and_eval_inputs_unchanged():
    # _build_description never read the removed metadata keys, so the
    # solver-facing description keeps its solve() contract. Eval inputs are
    # re-derived from _get_data() (pinned by the evaluate tests in
    # tests/test_text_classification.py).
    task = _stub_adapter().load_tasks()[0]
    assert "def solve(" in task.description
