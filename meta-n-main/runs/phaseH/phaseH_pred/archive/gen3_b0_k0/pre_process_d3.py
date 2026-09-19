desc = (getattr(task, "description", "") or "").lower()
meta = str(getattr(task, "metadata", "") or "").lower()
text = desc + " " + meta

if ("assortment" in text) or ("multinomial" in text) or ("logit" in text) or ("no-purchase" in text) or ("no purchase" in text):
    additional_context = """
Assortment optimization scoring is usually objective-quality based, not feasibility-only.

Use the single-segment helper `mnl_assortment(...)` only when the instance is the classic one-class MNL model:
    R(S) = sum_i v_i r_i / (v0 + sum_i v_i).

If the input has multiple customer classes, segments, scenarios, preference rows, arrival probabilities, or a matrix
of attraction weights, use `segmented_mnl_assortment(...)` instead. It directly optimizes the weighted multi-segment
expected revenue:
    sum_s alpha_s * sum_{i in S} v_{s,i} r_i / (v0_s + sum_{i in S} v_{s,i})
with optional cardinality and budget constraints. This is more robust than sorting by revenue/preference or applying
the single-segment threshold rule to averaged weights, because products can cannibalize demand differently across
segments.

Return the selected product indices in the format requested by the task, usually 1-based indices. If the benchmark
has extra constraints not represented by the helper, use the helper result as a starting solution, then apply small
feasibility-preserving add/drop/swap moves while recomputing the actual objective.
"""
else:
    additional_context = ""