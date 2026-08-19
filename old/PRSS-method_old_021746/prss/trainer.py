"""Alternating neural/spectral optimization with strict train-only Gram updates."""

from collections import defaultdict
from dataclasses import dataclass

import torch

from prss.losses import response_loss


@dataclass
class PRSSLosses:
  task: torch.Tensor
  response: torch.Tensor
  spectral: torch.Tensor
  unrestricted_diagnostic: torch.Tensor
  total: torch.Tensor


@dataclass
class PRSSTrainStep:
  losses: PRSSLosses
  readers_by_tau: dict
  spectral_updated: bool
  reader_norms: dict


class AlternatingPRSSTrainer:
  def __init__(self, system, executor, optimizer, task_loss_fn,
               response_task="binary", diagnostic_optimizer=None):
    self.system = system
    self.executor = executor
    self.optimizer = optimizer
    self.task_loss_fn = task_loss_fn
    self.response_task = response_task
    self.diagnostic_optimizer = diagnostic_optimizer
    self.step_index = 0

  def _forward_losses(self, trees, root_metadata, targets):
    executions = [self.executor.execute(tree) for tree in trees]
    task_outputs = torch.stack([execution.root_output for execution in executions], dim=0)
    task = self.task_loss_fn(task_outputs, targets)
    structured_logits = []
    unrestricted_logits = []
    repeated_targets = []
    readers_by_tau = defaultdict(list)
    spectral_terms = []

    for execution, metadata, target in zip(executions, root_metadata, targets):
      outputs = self.executor.outside_readers(execution, metadata)
      for output in outputs.values():
        structured_logits.append(output["structured_logits"])
        unrestricted_logits.append(output["unrestricted_logits"])
        repeated_targets.append(target)
        readers_by_tau[output["tau"]].append(output["matrix"])
        spectral_terms.append(self.system.state_spectral_loss(
          output["tau"], output["matrix"], output["candidate"]))

    structured_logits = torch.stack(structured_logits, dim=0)
    unrestricted_logits = torch.stack(unrestricted_logits, dim=0)
    repeated_targets = torch.stack(repeated_targets, dim=0)
    response = response_loss(structured_logits, repeated_targets, task=self.response_task)
    unrestricted = response_loss(unrestricted_logits, repeated_targets,
                                 task=self.response_task)
    spectral = torch.stack(spectral_terms).mean()
    readers_by_tau = {tau: torch.stack(values, dim=0)
                      for tau, values in readers_by_tau.items()}
    spectral_ready = any(
      int(state.spectral_updates.item()) > 0
      for state in self.system.quotients.states.values())
    total = (self.system.config.lambda_task * task +
             self.system.config.lambda_resp * response +
             (self.system.config.lambda_spec * spectral if spectral_ready else 0.0))
    return PRSSLosses(task, response, spectral, unrestricted, total), readers_by_tau

  def train_step(self, trees, root_metadata, targets):
    self.system.train()
    self.system.set_spectral_updates_allowed(True)
    losses, readers_by_tau = self._forward_losses(trees, root_metadata, targets)
    self.optimizer.zero_grad()
    losses.total.backward()
    self.optimizer.step()

    # Train the diagnostic comparator separately; its inputs were detached in the executor.
    if self.diagnostic_optimizer is not None:
      self.diagnostic_optimizer.zero_grad()
      losses.unrestricted_diagnostic.backward()
      self.diagnostic_optimizer.step()

    self.system.update_spectral_statistics(readers_by_tau)
    spectral_updated = self.system.maybe_spectral_update(self.step_index)
    result = PRSSTrainStep(
      losses=losses,
      readers_by_tau=readers_by_tau,
      spectral_updated=spectral_updated,
      reader_norms=self.system.reader_norms(readers_by_tau),
    )
    self.step_index += 1
    return result

  @torch.no_grad()
  def evaluate(self, trees, root_metadata, targets):
    self.system.eval()
    self.system.set_spectral_updates_allowed(False)
    losses, _ = self._forward_losses(trees, root_metadata, targets)
    return losses
