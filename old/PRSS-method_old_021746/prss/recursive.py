"""Generic compressed-first recursive executor and training-only outside pass."""

from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch

from prss.state import RecursiveOccurrence, RecursiveTrace


@dataclass
class TreeNode:
  node_id: int
  tau: str
  local_features: torch.Tensor
  children: List["TreeNode"] = field(default_factory=list)
  child_relations: List[int] = field(default_factory=list)
  child_delta_t: List[float] = field(default_factory=list)
  metadata: Dict[str, Any] = field(default_factory=dict)

  def __post_init__(self):
    if self.children:
      if not self.child_relations:
        self.child_relations = [0] * len(self.children)
      if not self.child_delta_t:
        self.child_delta_t = [0.0] * len(self.children)
    if len(self.child_relations) != len(self.children):
      raise ValueError("Each child needs one relation id")
    if len(self.child_delta_t) != len(self.children):
      raise ValueError("Each child needs one time delta")


@dataclass
class RecursiveExecution:
  trace: RecursiveTrace
  root_id: int
  root_quotient: torch.Tensor
  root_output: torch.Tensor


class CompressedRecursiveExecutor:
  """Host-agnostic executor: the host candidate function never receives a child candidate h."""
  def __init__(self, prss_system, host_candidate_fn, host_readout_fn):
    self.prss = prss_system
    self.host_candidate_fn = host_candidate_fn
    self.host_readout_fn = host_readout_fn

  def execute(self, root):
    trace = RecursiveTrace()

    def bottom_up(node):
      child_occurrences = [bottom_up(child) for child in node.children]
      # This list is deliberately quotient-only. Exposing candidate/raw here is an API violation.
      compressed_children = [occurrence.state.quotient for occurrence in child_occurrences]
      raw = self.host_candidate_fn(node, compressed_children)
      state = self.prss.make_state(node.tau, raw)
      occurrence = RecursiveOccurrence(
        occurrence_id=node.node_id,
        tau=node.tau,
        state=state,
        local_features=node.local_features,
        children=[child.occurrence_id for child in child_occurrences],
        child_relations=list(node.child_relations),
        child_delta_t=list(node.child_delta_t),
        metadata=dict(node.metadata),
      )
      trace.add(occurrence)
      return occurrence

    root_occurrence = bottom_up(root)
    trace.roots = [root_occurrence.occurrence_id]
    root_output = self.host_readout_fn(root_occurrence.state.quotient, root)
    return RecursiveExecution(trace, root_occurrence.occurrence_id,
                              root_occurrence.state.quotient, root_output)

  def outside_readers(self, execution, root_metadata):
    trace = execution.trace
    root = trace.occurrences[execution.root_id]
    contexts = {
      execution.root_id: self.prss.outside.root_context(root_metadata, root.tau)
    }
    queue = [execution.root_id]
    while queue:
      parent_id = queue.pop(0)
      parent = trace.occurrences[parent_id]
      parent_context = contexts[parent_id]
      for position, child_id in enumerate(parent.children):
        child = trace.occurrences[child_id]
        siblings_by_type = {}
        for sibling_id in parent.children:
          if sibling_id == child_id:
            continue
          sibling = trace.occurrences[sibling_id]
          siblings_by_type.setdefault(sibling.tau, []).append(sibling.state.candidate)
        sibling_tensors = {
          tau: torch.stack(values, dim=0) for tau, values in siblings_by_type.items()
        }
        sibling_summary = self.prss.outside.summarize_siblings(
          sibling_tensors, parent_context)
        contexts[child_id] = self.prss.outside.child_context(
          parent_outside=parent_context,
          parent_local=parent.local_features,
          relation_ids=parent.child_relations[position],
          delta_t=parent.child_delta_t[position],
          sibling_summary=sibling_summary,
          child_type=child.tau,
        )
        queue.append(child_id)

    outputs = {}
    for occurrence_id, occurrence in trace.occurrences.items():
      context = contexts[occurrence_id]
      structured, matrix, bias = self.prss.structured_read(
        occurrence.tau, context, occurrence.state.candidate)
      # Comparator cannot change Phi/outside/main branch.
      unrestricted = self.prss.unrestricted_read(
        occurrence.tau, context.detach(), occurrence.state.candidate.detach())
      outputs[occurrence_id] = {
        "tau": occurrence.tau,
        "context": context,
        "matrix": matrix,
        "bias": bias,
        "structured_logits": structured,
        "unrestricted_logits": unrestricted,
        "candidate": occurrence.state.candidate,
      }
    return outputs

