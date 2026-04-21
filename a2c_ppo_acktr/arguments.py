import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--algo', default='a2c', help='algorithm to use: a2c | ppo | acktr | kafe | kafe_shared')
    parser.add_argument(
        '--gail',
        action='store_true',
        default=False,
        help='do imitation learning with gail')
    parser.add_argument(
        '--gail-experts-dir',
        default='./gail_experts',
        help='directory that contains expert demonstrations for gail')
    parser.add_argument(
        '--gail-batch-size',
        type=int,
        default=128,
        help='gail batch size (default: 128)')
    parser.add_argument(
        '--gail-epoch', type=int, default=5, help='gail epochs (default: 5)')
    parser.add_argument(
        '--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--use-gae',
        action='store_true',
        default=False,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.01,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--num-processes',
        type=int,
        default=16,
        help='how many training CPU processes to use (default: 16)')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=5,
        help='number of forward steps in A2C (default: 5)')
    parser.add_argument(
        '--ppo-epoch',
        type=int,
        default=4,
        help='number of ppo epochs (default: 4)')
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=32,
        help='number of batches for ppo (default: 32)')
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.2,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--kafe-damping',
        type=float,
        default=1e-2,
        help='KAFE damping coefficient (default: 1e-2)')
    parser.add_argument(
        '--kafe-max-step-size',
        type=float,
        default=0.05,
        help='maximum KAFE actor step size (default: 0.05)')
    parser.add_argument(
        '--kafe-target-kl',
        type=float,
        default=0.01,
        help='target KL for KAFE trust-region scaling (default: 0.01)')
    parser.add_argument(
        '--kafe-kl-clip',
        type=float,
        default=None,
        help='ACKTR-style KL clip for scaling the KAFE preconditioned step (default: disabled)')
    parser.add_argument(
        '--kafe-fisher-clip',
        type=float,
        default=None,
        help='clip the KAFE update using v^T F v from a Fisher-vector product (default: disabled)')
    parser.add_argument(
        '--kafe-kernel-num-anchors',
        type=int,
        default=16,
        help='number of kernel anchors used by KAFE (default: 16)')
    parser.add_argument(
        '--kafe-kernel-sigma',
        type=float,
        default=1.0,
        help='RBF kernel sigma used by KAFE (default: 1.0)')
    parser.add_argument(
        '--kafe-statistic',
        default='logp',
        help='KAFE policy statistic: logits | probs | logp | score')
    parser.add_argument(
        '--kafe-critic-lr',
        type=float,
        default=None,
        help='critic learning rate for KAFE (default: use --lr)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=100,
        help='save interval, one save per n updates (default: 100)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=None,
        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=10e6,
        help='number of environment steps to train (default: 10e6)')
    parser.add_argument(
        '--env-name',
        default='PongNoFrameskip-v4',
        help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument(
        '--procgen-distribution-mode',
        default='easy',
        help='Procgen distribution_mode when --env-name uses procgen-<name> (default: easy)')
    parser.add_argument(
        '--procgen-num-levels',
        type=int,
        default=0,
        help='Procgen num_levels for training; 0 means unlimited levels (default: 0)')
    parser.add_argument(
        '--procgen-start-level',
        type=int,
        default=0,
        help='Procgen start_level for training (default: 0)')
    parser.add_argument(
        '--procgen-eval-num-levels',
        type=int,
        default=None,
        help='Procgen num_levels for evaluation (default: use training value)')
    parser.add_argument(
        '--procgen-eval-start-level',
        type=int,
        default=None,
        help='Procgen start_level for evaluation (default: use training value)')
    parser.add_argument(
        '--log-dir',
        default='/tmp/gym/',
        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument(
        '--save-dir',
        default='./trained_models/',
        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument(
        '--use-proper-time-limits',
        action='store_true',
        default=False,
        help='compute returns taking into account time limits')
    parser.add_argument(
        '--recurrent-policy',
        action='store_true',
        default=False,
        help='use a recurrent policy')
    parser.add_argument(
        '--use-linear-lr-decay',
        action='store_true',
        default=False,
        help='use a linear schedule on the learning rate')
    parser.add_argument(
        '--use-wandb',
        action='store_true',
        default=False,
        help='enable Weights & Biases logging')
    parser.add_argument(
        '--wandb-project',
        default='pytorch-a2c-ppo-acktr-gail',
        help='wandb project name')
    parser.add_argument(
        '--wandb-entity',
        default=None,
        help='wandb entity/team name (default: None)')
    parser.add_argument(
        '--wandb-name',
        default=None,
        help='wandb run name (default: auto-generated by wandb)')
    parser.add_argument(
        '--wandb-group',
        default=None,
        help='wandb group name (default: None)')
    parser.add_argument(
        '--wandb-tags',
        default='',
        help='comma-separated wandb tags, e.g. atari,ppo,debug')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    assert args.algo in ['a2c', 'ppo', 'acktr', 'kafe', 'kafe_shared']
    if args.recurrent_policy:
        assert args.algo in ['a2c', 'ppo'], \
            'Recurrent policy is not implemented for ACKTR/KAFE'

    return args
