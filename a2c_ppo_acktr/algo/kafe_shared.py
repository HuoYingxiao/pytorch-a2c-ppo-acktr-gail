import math

import torch
import torch.optim as optim


class KAFEShared():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 damping=1e-2,
                 max_step_size=0.05,
                 target_kl=0.01,
                 kl_clip=None,
                 kernel_num_anchors=16,
                 kernel_sigma=1.0,
                 statistic='logits',
                 critic_lr=None):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.lr = lr
        self.schedule_lr = critic_lr if critic_lr is not None else lr
        if self.schedule_lr is None:
            self.schedule_lr = 1.0

        self.damping = damping
        self.max_step_size = max_step_size
        self.target_kl = target_kl
        self.kl_clip = kl_clip if kl_clip is not None else target_kl
        self.kernel_num_anchors = kernel_num_anchors
        self.kernel_sigma = kernel_sigma
        self.statistic = 'logp'

        self.params = self._dedup_params(actor_critic.parameters())
        if len(self.params) == 0:
            raise ValueError('KAFE requires trainable parameters.')

        self.optimizer = optim.Adam(self.params, lr=self.schedule_lr, eps=eps)
        self._initial_schedule_lr = max(self.schedule_lr, 1e-12)

    def _dedup_params(self, params):
        seen = set()
        unique = []
        for param in params:
            if param is None or not param.requires_grad:
                continue
            if id(param) in seen:
                continue
            seen.add(id(param))
            unique.append(param)
        return unique

    def _flat(self, tensors):
        if len(tensors) == 0:
            device = next(self.actor_critic.parameters()).device
            return torch.zeros(0, device=device)
        return torch.cat([tensor.reshape(-1) for tensor in tensors])

    def _grads_or_zeros(self, grads, params):
        return [
            grad.detach() if grad is not None else torch.zeros_like(param)
            for grad, param in zip(grads, params)
        ]

    def _add_flat_(self, params, delta, alpha=1.0):
        offset = 0
        for param in params:
            numel = param.numel()
            param.add_(alpha * delta[offset:offset + numel].view_as(param))
            offset += numel

    def _clone_params(self, params):
        return [param.detach().clone() for param in params]

    def _restore_params_(self, params, saved_params):
        for param, saved in zip(params, saved_params):
            param.copy_(saved)

    def _categorical_kl_from_logits(self, old_logits, new_logits):
        old_log_probs = torch.log_softmax(old_logits, dim=-1)
        new_log_probs = torch.log_softmax(new_logits, dim=-1)
        old_probs = old_log_probs.exp()
        return (old_probs * (old_log_probs - new_log_probs)).sum(dim=-1).mean()

    def _sample_anchor_stats(self, stats):
        num_anchors = min(self.kernel_num_anchors, stats.size(0))
        indices = torch.randperm(stats.size(0), device=stats.device)[:num_anchors]
        return stats.detach()[indices].clone()

    def _kernel_features(self, stats, anchors):
        diff = stats.unsqueeze(1) - anchors.unsqueeze(0)
        sq_dist = (diff * diff).sum(dim=-1)
        sigma = self.kernel_sigma
        if sigma <= 0:
            sigma = sq_dist.detach().mean().sqrt().item()
        sigma2 = max(sigma * sigma, 1e-12)
        return torch.exp(-0.5 * sq_dist / sigma2)

    def _feature_covariance(self, features):
        centered = features - features.mean(dim=0, keepdim=True)
        return centered.t().mm(centered) / max(features.size(0), 1)

    def _build_jacobian(self, features, params):
        mean_features = features.mean(dim=0)
        rows = []
        for idx in range(mean_features.numel()):
            retain_graph = idx != mean_features.numel() - 1
            grads = torch.autograd.grad(mean_features[idx],
                                        params,
                                        retain_graph=retain_graph,
                                        create_graph=False,
                                        allow_unused=True)
            rows.append(self._flat(self._grads_or_zeros(grads, params)))
        return torch.stack(rows, dim=0)

    def _solve_direction(self, jacobian, cov_features, grad_flat):
        damping = max(self.damping, 1e-8)
        solve_dtype = torch.float64 if grad_flat.dtype == torch.float32 else grad_flat.dtype
        j64 = jacobian.to(solve_dtype)
        c64 = cov_features.to(solve_dtype)
        g64 = grad_flat.to(solve_dtype)

        system = c64 + (1.0 / damping) * (j64 @ j64.t())
        system = 0.5 * (system + system.t())
        eye = torch.eye(system.size(0), device=system.device, dtype=system.dtype)
        system = system + 1e-6 * eye
        u = j64 @ g64

        try:
            chol = torch.linalg.cholesky(system)
            y = torch.cholesky_solve(u.unsqueeze(-1), chol).squeeze(-1)
        except RuntimeError:
            y = torch.linalg.pinv(system) @ u

        delta = (1.0 / damping) * g64 - \
            (1.0 / (damping * damping)) * (j64.t() @ y)
        delta = delta.to(grad_flat.dtype)
        fisher_quad = torch.dot(grad_flat, delta).clamp_min(1e-12)
        return delta, fisher_quad

    def _compute_value_loss(self, values, value_preds_batch, return_batch):
        if self.use_clipped_value_loss:
            value_pred_clipped = value_preds_batch + \
                (values - value_preds_batch).clamp(-self.clip_param,
                                                   self.clip_param)
            value_losses = (values - return_batch).pow(2)
            value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
            return 0.5 * torch.max(value_losses, value_losses_clipped).mean()

        return 0.5 * (return_batch - values).pow(2).mean()

    def _compute_joint_stats(self, dist, actions_batch, values):
        policy_stat = dist.log_probs(actions_batch)

        value_noise = torch.randn_like(values)
        sample_values = values + value_noise
        value_stat = values - sample_values.detach()

        return torch.cat([policy_stat, value_stat], dim=-1)

    def _compute_losses(self,
                        obs_batch,
                        recurrent_hidden_states_batch,
                        actions_batch,
                        value_preds_batch,
                        return_batch,
                        masks_batch,
                        old_action_log_probs_batch,
                        adv_targ):
        values, dist, _, _ = self.actor_critic.get_dist(
            obs_batch, recurrent_hidden_states_batch, masks_batch)
        action_log_probs = dist.log_probs(actions_batch)
        dist_entropy = dist.entropy().mean()

        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = ratio * adv_targ
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                            1.0 + self.clip_param) * adv_targ
        action_loss = -torch.min(surr1, surr2).mean()
        value_loss = self._compute_value_loss(values, value_preds_batch,
                                              return_batch)
        total_loss = value_loss * self.value_loss_coef + action_loss - \
            dist_entropy * self.entropy_coef

        return {
            'values': values,
            'dist': dist,
            'action_log_probs': action_log_probs,
            'dist_entropy': dist_entropy,
            'ratio': ratio,
            'action_loss': action_loss,
            'value_loss': value_loss,
            'total_loss': total_loss
        }

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        approx_kl_epoch = 0
        log_prob_epoch = 0
        ratio_mean_epoch = 0
        grad_norm_epoch = 0
        actor_step_size_epoch = 0
        fisher_quad_epoch = 0
        actual_kl_epoch = 0
        precond_scale_epoch = 0

        current_lr = self.optimizer.param_groups[0]['lr']
        lr_scale = current_lr / self._initial_schedule_lr

        for _ in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                    value_preds_batch, return_batch, masks_batch, \
                    old_action_log_probs_batch, adv_targ = sample

                self.optimizer.zero_grad()
                self.actor_critic.zero_grad()

                losses = self._compute_losses(obs_batch,
                                              recurrent_hidden_states_batch,
                                              actions_batch,
                                              value_preds_batch,
                                              return_batch,
                                              masks_batch,
                                              old_action_log_probs_batch,
                                              adv_targ)
                grads = torch.autograd.grad(losses['total_loss'],
                                            self.params,
                                            retain_graph=True,
                                            create_graph=False,
                                            allow_unused=True)
                grad_flat = self._flat(self._grads_or_zeros(grads, self.params))
                grad_norm = grad_flat.norm().item()

                stats = self._compute_joint_stats(losses['dist'],
                                                  actions_batch,
                                                  losses['values'])
                anchors = self._sample_anchor_stats(stats)
                features = self._kernel_features(stats, anchors)
                cov_features = self._feature_covariance(features.detach())
                jacobian = self._build_jacobian(features, self.params)

                old_logits = losses['dist'].logits.detach()
                old_total_loss = float(losses['total_loss'].detach())
                delta, fisher_quad = self._solve_direction(jacobian,
                                                           cov_features,
                                                           grad_flat)
                saved_params = self._clone_params(self.params)

                if self.target_kl > 0:
                    trust_region_scale = math.sqrt(
                        2.0 * self.target_kl / fisher_quad.item())
                    step_size = min(self.max_step_size, trust_region_scale)
                else:
                    step_size = self.max_step_size

                step_size *= lr_scale

                precond_scale = 1.0
                if self.kl_clip is not None and self.kl_clip > 0:
                    precond_quad = step_size * step_size * fisher_quad.item()
                    precond_scale = min(
                        1.0, math.sqrt(self.kl_clip / max(precond_quad, 1e-12)))
                    step_size *= precond_scale

                actual_kl = 0.0
                accepted = False
                with torch.no_grad():
                    trial_step_size = step_size
                    for _ in range(10):
                        self._restore_params_(self.params, saved_params)
                        self._add_flat_(self.params, -delta, alpha=trial_step_size)

                        trial_losses = self._compute_losses(
                            obs_batch,
                            recurrent_hidden_states_batch,
                            actions_batch,
                            value_preds_batch,
                            return_batch,
                            masks_batch,
                            old_action_log_probs_batch,
                            adv_targ)
                        actual_kl = float(self._categorical_kl_from_logits(
                            old_logits, trial_losses['dist'].logits))
                        new_total_loss = float(trial_losses['total_loss'])

                        kl_ok = self.target_kl <= 0 or actual_kl <= 1.5 * self.target_kl
                        loss_ok = math.isfinite(new_total_loss) and \
                            new_total_loss <= old_total_loss + 1e-6

                        if kl_ok and loss_ok:
                            accepted = True
                            step_size = trial_step_size
                            losses = trial_losses
                            break

                        trial_step_size *= 0.5

                if not accepted:
                    with torch.no_grad():
                        self._restore_params_(self.params, saved_params)
                    step_size = 0.0
                    actual_kl = 0.0

                value_loss_epoch += losses['value_loss'].item()
                action_loss_epoch += losses['action_loss'].item()
                dist_entropy_epoch += losses['dist_entropy'].item()
                approx_kl_epoch += (old_action_log_probs_batch -
                                    losses['action_log_probs']).mean().item()
                log_prob_epoch += losses['action_log_probs'].mean().item()
                ratio_mean_epoch += losses['ratio'].mean().item()
                grad_norm_epoch += grad_norm
                actor_step_size_epoch += step_size
                fisher_quad_epoch += fisher_quad.item()
                actual_kl_epoch += actual_kl
                precond_scale_epoch += precond_scale

        num_updates = self.ppo_epoch * self.num_mini_batch
        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        approx_kl_epoch /= num_updates
        log_prob_epoch /= num_updates
        ratio_mean_epoch /= num_updates
        grad_norm_epoch /= num_updates
        actor_step_size_epoch /= num_updates
        fisher_quad_epoch /= num_updates
        actual_kl_epoch /= num_updates
        precond_scale_epoch /= num_updates

        diagnostics = {
            'value_loss': value_loss_epoch,
            'action_loss': action_loss_epoch,
            'dist_entropy': dist_entropy_epoch,
            'grad_norm': grad_norm_epoch,
            'critic_grad_norm': None,
            'log_prob': log_prob_epoch,
            'approx_kl': approx_kl_epoch,
            'actual_kl': actual_kl_epoch,
            'ratio_mean': ratio_mean_epoch,
            'actor_step_size': actor_step_size_epoch,
            'precond_scale': precond_scale_epoch,
            'fisher_quad': fisher_quad_epoch
        }

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, diagnostics
