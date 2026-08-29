"""Rewrite the loop for macro-group timing + kf_on fast path + rank
normalization (eighth review, sections C/D/A)."""
import re

p = 'src/rpbe/training/jodie_loop.py'
s = open(p, encoding='utf-8').read()

# ---- 1. constructor: two optimizers, kf_on separation, tau_coeff ----
s = s.replace('''    def __init__(self, *, tgn, decoder, optimizer, device, batch_size,
                 n_neighbors, grad_clip, monitor, seed, finetune_host=False,
                 selection_metric="auc", adapter=None, cut_builder=None,
                 fixed_maps=None, rpbe_cfg=None, trace_roots=8,
                 trace_mode="evenly_spaced", train_eval_auc=False):
        self.tgn = tgn
        self.decoder = decoder
        self.optimizer = optimizer
        self.device = device''',
'''    def __init__(self, *, tgn, decoder, repr_optimizer, head_optimizer,
                 device, batch_size, n_neighbors, grad_clip, monitor, seed,
                 finetune_host=False, selection_metric="auc", adapter=None,
                 cut_builder=None, fixed_maps=None, rpbe_cfg=None,
                 trace_roots=8, trace_mode="evenly_spaced",
                 train_eval_auc=False):
        self.tgn = tgn
        self.decoder = decoder
        # Representation parameters (host + compressor, everything that can
        # change a cut's z) update once per macro-group; the head updates
        # every batch.  Both groups receive gradients from every backward.
        self.repr_optimizer = repr_optimizer
        self.head_optimizer = head_optimizer
        self.repr_params = list(repr_optimizer.param_groups[0]["params"])
        self.head_params = list(head_optimizer.param_groups[0]["params"])
        self.device = device''')

# ---- 2. component_on / kf_on separation ----
s = s.replace('''        self.rpbe_on = bool(adapter is not None and cut_builder is not None
                            and fixed_maps is not None
                            and rpbe_cfg is not None)''',
'''        # Eighth review D: lambda=0 is a TRUE task-only fast path — the
        # compressor still shapes the forward (TGN + Gamma, task only), but
        # no tracing, no cuts, no P projection, no window updates.
        self.component_on = bool(adapter is not None
                                 and fixed_maps is not None
                                 and rpbe_cfg is not None)
        self.kf_on = bool(self.component_on and self.lambda_kf > 0.0
                          and cut_builder is not None)
        self.rpbe_on = self.component_on''')

# ---- 3. tau_coeff (rank normalization) ----
s = s.replace('''        self.kf_window = (KFLaggedWindow(''',
'''        # Eighth review A: per-interface rank normalization coefficients.
        # J_norm = sum_tau alpha_tau J_tau / min(d_tau, m) / sum alpha_tau
        # lives in ~[0, 1]; the denominator is FIXED by the configured
        # interfaces, never by which taus happen to activate in a batch.
        self._tau_coeff = {}
        if self.kf_on:
            alpha_sum = float(sum(
                rpbe_cfg.alpha(tau) for tau in kf_dims))
            for tau in kf_dims:
                self._tau_coeff[tau] = rpbe_cfg.alpha(tau) / (
                    min(int(rpbe_cfg.state_dims[tau]),
                        int(fixed_maps.m)) * alpha_sum)
        self.kf_window = (KFLaggedWindow(''')

# ---- 4. clip helpers for per-group clipping ----
s = s.replace('''    def _clip_all_groups(self):
        if self.grad_clip <= 0:
            return
        params = [param for group in self.optimizer.param_groups
                  for param in group["params"] if param.grad is not None]
        if params:
            torch.nn.utils.clip_grad_norm_(
                params, max_norm=self.grad_clip, error_if_nonfinite=True)''',
'''    def _clip(self, params):
        if self.grad_clip <= 0:
            return
        live = [p for p in params if p.grad is not None]
        if live:
            torch.nn.utils.clip_grad_norm_(
                live, max_norm=self.grad_clip, error_if_nonfinite=True)''')

# ---- 5. _consume_kf with tau_coeff and normalized monitoring ----
s = s.replace('''    def _consume_kf(self, cuts, step):
        scores, surrogates, diagnostics, cold, refreshed = \\
            self.kf_window.step(cuts)
        weighted_score = float(sum(
            self.rpbe_cfg.alpha(tau) * score
            for tau, score in scores.items()))
        if surrogates:
            auxiliary = -self.lambda_kf * sum(
                self.rpbe_cfg.alpha(tau) * value
                for tau, value in surrogates.items())
        else:
            auxiliary = torch.zeros((), device=self.device)

        if scores:
            bounds = {tau: min(int(self.rpbe_cfg.state_dims[tau]),
                               int(self.fixed_maps.m))
                      for tau in scores}
            self.monitor.validate_kf(scores, bounds, step)
        for tau, diag in diagnostics.items():
            if diag.get("failed"):
                self.monitor.alert(
                    "warning", "kf_reference_failed",
                    "{} {} scale_z={:.3e} scale_p={:.3e} trees={}".format(
                        tau, diag["failed"],
                        diag.get("scale_z", float("nan")),
                        diag.get("scale_p", float("nan")),
                        diag["M_unique_trees"]),
                    step=step, interface=tau)
        return (weighted_score, scores, auxiliary, diagnostics,
                set(cold), len(refreshed))''',
'''    def _consume_kf(self, cuts, step):
        scores, surrogates, cold = self.kf_window.consume(cuts)
        raw_score = float(sum(
            self.rpbe_cfg.alpha(tau) * score
            for tau, score in scores.items()))
        norm_score = float(sum(
            self._tau_coeff[tau] * score
            for tau, score in scores.items()))
        if surrogates:
            auxiliary = -self.lambda_kf * sum(
                self._tau_coeff[tau] * value
                for tau, value in surrogates.items())
        else:
            auxiliary = torch.zeros((), device=self.device)

        if scores:
            bounds = {tau: min(int(self.rpbe_cfg.state_dims[tau]),
                               int(self.fixed_maps.m))
                      for tau in scores}
            self.monitor.validate_kf(scores, bounds, step)
        return raw_score, norm_score, scores, auxiliary, set(cold)''')

# ---- 6. train_epoch macro-group timing ----
old_epoch_start = s.index('    # ----------------------------------------------------------------- training\n    def train_epoch')
old_epoch_end = s.index('    # --------------------------------------------------------------- evaluation')
new_epoch = '''    # ----------------------------------------------------------------- training
    def train_epoch(self, epoch: int, global_step: int, train: object) -> Dict:
        """One chronological epoch.

        Macro-group timing (eighth review C): representation parameters
        (host + compressor) are frozen within a macro-group and update once
        when the KF pending windows close (or, in the task-only fast path,
        every fixed K batches so both protocols share the same optimizer
        cadence).  The head updates every batch.  One query, one backward,
        one head step per batch — no replay, no cross-batch graphs.
        """
        self.reset_memory()
        self.kf_window.reset(clear_reference=True)   # eighth review C
        self.tgn.train(self.finetune_host)
        self.decoder.train()

        # Fixed group length for the task-only fast path (same cadence as
        # the KF path: ceil(min_abs / trace_roots) batches).
        fixed_group = max(1, math.ceil(
            self.rpbe_cfg.kf_min_abs / max(1, self.trace_roots))) \\
            if self.component_on else 1

        total_task = 0.0
        total_raw = 0.0
        total_norm = 0.0
        kf_sum = {}
        kf_count_tau = {}
        cold_types = set()
        refreshes = 0
        aux_batches = 0
        no_cuts_alerted = False
        param_version = 0
        repr_group_active = False
        group_batch_count = 0
        train_probs, train_labels = [], []
        num_batch = math.ceil(len(train.sources) / self.batch_size)

        for batch_index in range(num_batch):
            start = batch_index * self.batch_size
            stop = min(len(train.sources), (batch_index + 1) * self.batch_size)
            sources = train.sources[start:stop]
            destinations = train.destinations[start:stop]
            timestamps = train.timestamps[start:stop]
            edge_idxs = train.edge_idxs[start:stop]
            labels_np = train.labels[start:stop]
            labels_t = torch.from_numpy(labels_np).float().to(self.device)

            self.head_optimizer.zero_grad(set_to_none=True)
            if not repr_group_active:
                self.repr_optimizer.zero_grad(set_to_none=True)
                self.kf_window.begin_group(param_version, epoch)
                repr_group_active = True
                group_batch_count = 0

            trace_rows = []
            if self.kf_on:
                trace_rows = select_trace_rows(
                    labels_np, self.trace_roots, self.seed, global_step,
                    self.trace_mode)
                self.adapter.set_trace_source_rows(trace_rows)
            elif self.adapter is not None:
                self.adapter.clear_trace()

            src_emb, _, _ = self._full_official_embedding_call(
                sources, destinations, timestamps, edge_idxs,
                grad_enabled=(self.finetune_host or self.component_on))
            prediction = self.decoder(src_emb).sigmoid()
            task_loss = F.binary_cross_entropy(prediction, labels_t)
            auxiliary = torch.zeros((), device=self.device)
            raw_score = 0.0
            norm_score = 0.0

            if self.kf_on and trace_rows and self.adapter.trace is not None:
                cuts = self.cut_builder.build(
                    self.adapter.trace, batch_seed=global_step)
                if cuts:
                    (raw_score, norm_score, scores, auxiliary,
                     cold) = self._consume_kf(cuts, global_step)
                    cold_types.update(cold)
                    if auxiliary.requires_grad:
                        aux_batches += 1
                    for tau, score in scores.items():
                        kf_sum[tau] = kf_sum.get(tau, 0.0) + score
                        kf_count_tau[tau] = kf_count_tau.get(tau, 0) + 1
                elif not no_cuts_alerted:
                    self.monitor.alert(
                        "warning", "kf_no_future_rows",
                        "traced cuts have no strictly-future train outcomes",
                        step=global_step)
                    no_cuts_alerted = True

            loss = task_loss + auxiliary
            self.monitor.validate_losses({
                "task": float(task_loss.detach()),
                "kf_score": raw_score,
                "kf_normalized": norm_score,
                "main_total": float(loss.detach()),
            }, global_step)
            loss.backward()
            self._clip(self.head_params)
            self.head_optimizer.step()
            if self.tgn.use_memory and \\
                    (self.finetune_host or self.component_on):
                self.tgn.memory.detach_memory()

            total_task += float(task_loss.detach())
            total_raw += raw_score
            total_norm += norm_score
            train_probs.append(prediction.detach().cpu().numpy())
            train_labels.append(labels_np)
            group_batch_count += 1
            global_step += 1

            # Macro-group close: KF path closes on pending readiness; the
            # task-only fast path uses the fixed cadence.
            close_now = False
            if self.kf_on and self.kf_window.all_pending_ready():
                close_now = True
            elif not self.kf_on and self.component_on \\
                    and group_batch_count >= fixed_group:
                close_now = True
            if close_now:
                if self.kf_on:
                    diagnostics, refreshed = self.kf_window.close_group()
                    refreshes += len(refreshed)
                    for tau, diag in diagnostics.items():
                        if diag.get("failed"):
                            self.monitor.alert(
                                "warning", "kf_reference_failed",
                                "{} {} scale_z={:.3e} scale_p={:.3e} "
                                "trees={}".format(
                                    tau, diag["failed"],
                                    diag.get("scale_z", float("nan")),
                                    diag.get("scale_p", float("nan")),
                                    diag["M_unique_trees"]),
                                step=global_step, interface=tau)
                for p in self.repr_params:
                    if p.grad is not None:
                        p.grad.div_(float(group_batch_count))
                self._clip(self.repr_params)
                self.repr_optimizer.step()
                param_version += 1
                if self.kf_on:
                    self.kf_window.commit_reference()
                repr_group_active = False

        # Epoch drain: update the representation parameters with the
        # accumulated task gradients of the unfinished group.
        if repr_group_active:
            for p in self.repr_params:
                if p.grad is not None:
                    p.grad.div_(float(max(1, group_batch_count)))
            self._clip(self.repr_params)
            self.repr_optimizer.step()
            self.repr_optimizer.zero_grad(set_to_none=True)

        train_metrics = metric_bundle(np.concatenate(train_labels),
                                      np.concatenate(train_probs))
        train_metrics["online_auc"] = train_metrics.pop("auc")
        train_metrics["online_ap"] = train_metrics.pop("ap")
        if self.train_eval_auc:
            backup = self.tgn.memory.backup_memory() \\
                if self.tgn.use_memory else None
            eval_row = self.evaluate_split(train, reset=True)
            if backup is not None:
                self.tgn.memory.restore_memory(backup)
            train_metrics["eval_auc"] = eval_row["auc"]
            train_metrics["eval_ap"] = eval_row["ap"]

        kf_out = None
        if kf_count_tau and self.rpbe_cfg is not None:
            means = {tau: value / kf_count_tau[tau]
                     for tau, value in kf_sum.items()}
            kf_out = {
                "estimator": "one_pass_lagged_moment_adjoint",
                "J": means,
                "J_norm": {tau: value / min(
                    int(self.rpbe_cfg.state_dims[tau]),
                    int(self.fixed_maps.m)) for tau, value in means.items()},
                "skipped_types": sorted(cold_types),
                "kf_score": total_raw / max(num_batch, 1),
                "kf_normalized": total_norm / max(num_batch, 1),
                "kf_loss": -self.lambda_kf * total_norm / max(num_batch, 1),
                "p_cache_hits": getattr(self.fixed_maps, "_p_cache_hits", 0),
                "p_cache_misses": getattr(
                    self.fixed_maps, "_p_cache_misses", 0),
                "reference_refreshes": refreshes,
                "aux_batches": aux_batches,
            }
        return {
            "train_task_loss": total_task / max(num_batch, 1),
            "train_kf_score": total_raw / max(num_batch, 1),
            "train_kf_normalized": total_norm / max(num_batch, 1),
            "kf": kf_out,
            "train": train_metrics,
            "n_batches": num_batch,
            "global_step": global_step,
        }

'''
s = s[:old_epoch_start] + new_epoch + s[old_epoch_end:]
open(p, 'w', encoding='utf-8').write(s)
print('loop rewritten')
