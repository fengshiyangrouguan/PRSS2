# Provide targeted guidance only when the task description structurally looks like
# an assortment/revenue choice optimization problem. Do not modify the task object.
desc = (getattr(task, "description", "") or "").lower()
meta = str(getattr(task, "metadata", "") or "").lower()
text = desc + " " + meta

if ("assortment" in text) or ("multinomial" in text) or ("logit" in text) or ("no-purchase" in text) or ("no purchase" in text):
    additional_context = """
For assortment-style revenue problems, optimize the expected objective, not just feasibility.
If the model is multinomial-logit/MNL with revenues r_i, preference weights v_i, and no-purchase
weight v0, the expected revenue of offered set S is:

    R(S) = sum_{i in S} v_i*r_i / (v0 + sum_{i in S} v_i)

Do NOT simply sort by revenue or by preference weight: high-attraction low-revenue products can
cannibalize better items. Use the provided helper `mnl_assortment(...)` when the input resembles
an MNL assortment problem. It handles optional cardinality limits and returns 1-based product
indices plus the expected revenue. If the benchmark has additional constraints, use the helper's
selected set as a high-quality starting point, then run small add/drop/swap local search while
recomputing the true objective and preserving feasibility.
"""
else:
    additional_context = ""