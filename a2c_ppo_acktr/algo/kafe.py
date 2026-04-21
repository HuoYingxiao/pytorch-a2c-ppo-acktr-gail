import math

import torch
import torch.nn as nn
import torch.optim as optim


class KAFE():
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
                 fisher_clip=None,
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
        self.critic_lr = critic_lr if critic_lr is not None else lr
        self.damping = damping
        self.max_step_size = max_step_size
        self.target_kl = target_kl
        self.kl_clip = kl_clip
        self.fisher_clip = fisher_clip
        self.kernel_num_anchors = kernel_num_anchors
        self.kernel_sigma = kernel_sigma
        self.statistic = statistic

        shared_params, actor_params, critic_params = \
            actor_critic.get_kafe_param_groups()

        self.shared_params = self._dedup_params(shared_params)
        self.actor_params = self._dedup_params(actor_params)
        self.critic_params = self._dedup_params(critic_params)
        self.actor_group_params = self._dedup_params(
            self.shared_params + self.actor_params)

        if len(self.critic_params) == 0:
            raise ValueError('KAFE requires critic parameters.')

        self.optimizer = optim.Adam(self.critic_params,
                                    lr=self.critic_lr,
                                    eps=eps)

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
            return torch.zeros(0, device=self.actor_critic.base.critic_linear.weight.device)
        return torch.cat([tensor.reshape(-1) for tensor in tensors])

    def _grads_or_zeros(self, grads, params):
        return [
            grad.detach() if grad is not None else torch.zeros_like(param)
            for grad, param in zip(grads, params)
        ]

    def _grads_or_zeros_with_graph(self, grads, params):
        return [
            grad if grad is not None else torch.zeros_like(param)
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

    def _get_distribution_stats(self, dist, actions):
        if dist.__class__.__name__ != 'FixedCategorical':
            raise NotImplementedError(
                'KAFE is implemented for Atari/discrete action spaces in this repository.')

        if actions.dim() > 1:
            actions = actions.squeeze(-1)

        if self.statistic == 'logits':
            return dist.logits
        if self.statistic == 'probs':
            return dist.probs
        if self.statistic == 'logp':
            return dist.log_prob(actions).unsqueeze(-1)
        if self.statistic == 'score':
            one_hot = torch.nn.functional.one_hot(
                actions.long(), num_classes=dist.probs.size(-1)).float()
            return one_hot - dist.probs

        raise ValueError('Unsupported KAFE statistic: {}'.format(
            self.statistic))

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
        cov = centered.t().mm(centered) / max(features.size(0), 1)
        return cov

    def _build_jacobian(self, features, params):
        mean_features = features.mean(dim=0)
        rows = []
        for idx in range(mean_features.numel()):
            # Keep the forward graph alive for subsequent Fisher-vector products.
            grads = torch.autograd.grad(mean_features[idx],
                                        params,
                                        retain_graph=True,
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

    def _fisher_vector_product(self, dist, vector):
        kl = self._categorical_kl_from_logits(dist.logits.detach(), dist.logits)
        kl_grads = torch.autograd.grad(kl,
                                       self.actor_group_params,
                                       retain_graph=True,
                                       create_graph=True,
                                       allow_unused=True)
        flat_kl_grads = self._flat(
            self._grads_or_zeros_with_graph(kl_grads, self.actor_group_params))
        grad_vector_product = torch.dot(flat_kl_grads, vector)
        hvp = torch.autograd.grad(grad_vector_product,
                                  self.actor_group_params,
                                  retain_graph=True,
                                  create_graph=False,
                                  allow_unused=True)
        return self._flat(self._grads_or_zeros(hvp, self.actor_group_params))

    def _critic_step(self, obs_batch, recurrent_hidden_states_batch,
                     masks_batch, value_preds_batch, return_batch):
        values = self.actor_critic.get_value(obs_batch,
                                             recurrent_hidden_states_batch,
                                             masks_batch)
        value_loss = self._compute_value_loss(values, value_preds_batch,
                                              return_batch)

        self.optimizer.zero_grad()
        value_loss.backward()
        critic_grad_norm = 0.0
        if self.max_grad_norm is not None and self.max_grad_norm > 0:
            critic_grad_norm = float(nn.utils.clip_grad_norm_(
                self.critic_params, self.max_grad_norm))
        self.optimizer.step()
        return value_loss.detach(), critic_grad_norm

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
        actor_grad_norm_epoch = 0
        actor_step_size_epoch = 0
        critic_grad_norm_epoch = 0
        fisher_quad_epoch = 0
        actual_kl_epoch = 0
        precond_scale_epoch = 0
        fisher_scale_epoch = 0

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

                values, dist, _, _ = self.actor_critic.get_dist(
                    obs_batch, recurrent_hidden_states_batch, masks_batch)
                action_log_probs = dist.log_probs(actions_batch)
                dist_entropy = dist.entropy().mean()

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()
                actor_objective = action_loss - dist_entropy * self.entropy_coef

                value_loss = self._compute_value_loss(values,
                                                      value_preds_batch,
                                                      return_batch)

                if len(self.shared_params) > 0:
                    shared_grads = torch.autograd.grad(
                        actor_objective,
                        self.shared_params,
                        retain_graph=True,
                        create_graph=False,
                        allow_unused=True)
                else:
                    shared_grads = []

                actor_grads = torch.autograd.grad(actor_objective,
                                                  self.actor_params,
                                                  retain_graph=True,
                                                  create_graph=False,
                                                  allow_unused=True)

                grad_flat = self._flat(
                    self._grads_or_zeros(shared_grads, self.shared_params) +
                    self._grads_or_zeros(actor_grads, self.actor_params))
                actor_grad_norm = grad_flat.norm().item()

                stats = self._get_distribution_stats(dist, actions_batch)
                anchors = self._sample_anchor_stats(stats)
                features = self._kernel_features(stats, anchors)
                cov_features = self._feature_covariance(features.detach())
                jacobian = self._build_jacobian(features,
                                                self.actor_group_params)
                old_logits = dist.logits.detach()
                saved_actor_params = self._clone_params(self.actor_group_params)

                delta, fisher_quad = self._solve_direction(jacobian,
                                                           cov_features,
                                                           grad_flat)

                if self.target_kl > 0:
                    trust_region_scale = math.sqrt(
                        2.0 * self.target_kl / fisher_quad.item())
                    step_size = min(self.max_step_size, trust_region_scale)
                else:
                    step_size = self.max_step_size

                precond_scale = 1.0
                if self.kl_clip is not None and self.kl_clip > 0:
                    precond_quad = step_size * step_size * fisher_quad.item()
                    precond_scale = min(
                        1.0, math.sqrt(self.kl_clip / max(precond_quad, 1e-12)))
                    step_size *= precond_scale

                fisher_scale = 1.0
                if self.fisher_clip is not None and self.fisher_clip > 0:
                    step_dir = step_size * delta
                    fisher_vec = self._fisher_vector_product(dist, step_dir)
                    fisher_dir_quad = torch.dot(step_dir, fisher_vec).clamp_min(0.0)
                    fisher_scale = min(
                        1.0,
                        self.fisher_clip / max(fisher_dir_quad.item(), 1e-12))
                    step_size *= fisher_scale

                actual_kl = 0.0
                with torch.no_grad():
                    accepted = False
                    trial_step_size = step_size
                    for _ in range(10):
                        self._restore_params_(self.actor_group_params,
                                              saved_actor_params)
                        self._add_flat_(self.actor_group_params,
                                        -trial_step_size * delta)

                        _, new_dist, _, _ = self.actor_critic.get_dist(
                            obs_batch,
                            recurrent_hidden_states_batch,
                            masks_batch)
                        actual_kl = float(self._categorical_kl_from_logits(
                            old_logits, new_dist.logits))

                        if self.target_kl <= 0 or actual_kl <= 1.5 * self.target_kl:
                            accepted = True
                            step_size = trial_step_size
                            break

                        trial_step_size *= 0.5

                    if not accepted:
                        self._restore_params_(self.actor_group_params,
                                              saved_actor_params)
                        step_size = 0.0
                        actual_kl = 0.0

                value_loss, critic_grad_norm = self._critic_step(
                    obs_batch,
                    recurrent_hidden_states_batch,
                    masks_batch,
                    value_preds_batch,
                    return_batch)

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                approx_kl_epoch += (old_action_log_probs_batch -
                                    action_log_probs).mean().item()
                log_prob_epoch += action_log_probs.mean().item()
                ratio_mean_epoch += ratio.mean().item()
                actor_grad_norm_epoch += actor_grad_norm
                actor_step_size_epoch += step_size
                critic_grad_norm_epoch += critic_grad_norm
                fisher_quad_epoch += fisher_quad.item()
                actual_kl_epoch += actual_kl
                precond_scale_epoch += precond_scale
                fisher_scale_epoch += fisher_scale

        num_updates = self.ppo_epoch * self.num_mini_batch
        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        approx_kl_epoch /= num_updates
        log_prob_epoch /= num_updates
        ratio_mean_epoch /= num_updates
        actor_grad_norm_epoch /= num_updates
        actor_step_size_epoch /= num_updates
        critic_grad_norm_epoch /= num_updates
        fisher_quad_epoch /= num_updates
        actual_kl_epoch /= num_updates
        precond_scale_epoch /= num_updates
        fisher_scale_epoch /= num_updates

        diagnostics = {
            'value_loss': value_loss_epoch,
            'action_loss': action_loss_epoch,
            'dist_entropy': dist_entropy_epoch,
            'grad_norm': actor_grad_norm_epoch,
            'critic_grad_norm': critic_grad_norm_epoch,
            'log_prob': log_prob_epoch,
            'approx_kl': approx_kl_epoch,
            'actual_kl': actual_kl_epoch,
            'ratio_mean': ratio_mean_epoch,
            'actor_step_size': actor_step_size_epoch,
            'precond_scale': precond_scale_epoch,
            'fisher_scale': fisher_scale_epoch,
            'fisher_quad': fisher_quad_epoch
        }

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, diagnostics
