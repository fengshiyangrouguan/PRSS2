import torch
from torch.nn import functional as F

from prss.config import InterfaceSpec, PRSSConfig
from prss.recursive import CompressedRecursiveExecutor, TreeNode
from prss.system import PRSSSystem


def test_parent_receives_only_host_width_quotients_and_outside_excludes_self():
  torch.manual_seed(1)
  config = PRSSConfig(
    interfaces={"node": InterfaceSpec("node", raw_dim=4, candidate_dim=7, host_dim=2)},
    context_dim=5,
    root_metadata_dim=3,
    parent_local_dim=4,
    relation_count=3,
  )
  system = PRSSSystem(config)
  received_widths = []

  leaf_encoder = torch.nn.Linear(4, 4)
  parent_encoder = torch.nn.Linear(4 + 2 * 2, 4)

  def host_candidate(node, compressed_children):
    received_widths.extend(value.shape[-1] for value in compressed_children)
    if not compressed_children:
      return leaf_encoder(node.local_features)
    padded = compressed_children + [torch.zeros_like(compressed_children[0])] * (
      2 - len(compressed_children))
    return parent_encoder(torch.cat([node.local_features] + padded[:2], dim=-1))

  readout = torch.nn.Linear(2, 1)
  executor = CompressedRecursiveExecutor(system, host_candidate,
                                         lambda quotient, _: readout(quotient))
  left = TreeNode(1, "node", torch.randn(4))
  right = TreeNode(2, "node", torch.randn(4))
  root = TreeNode(3, "node", torch.randn(4), children=[left, right],
                  child_relations=[1, 2], child_delta_t=[1.0, 2.0])
  execution = executor.execute(root)
  outputs = executor.outside_readers(execution, torch.randn(3))

  assert received_widths == [2, 2]
  assert execution.root_quotient.shape[-1] == 2
  assert len(outputs) == 3
  # A child's outside state can depend on its sibling candidate, but never receives its own h
  # through the API: summarize_siblings is called with IDs excluding that child.
  assert outputs[1]["context"].shape[-1] == config.context_dim
  assert outputs[2]["context"].shape[-1] == config.context_dim

