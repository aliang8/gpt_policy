import math
import torch
import torch.nn as nn
from rl.modules.mlp import mlp
from rl.rl_utils import weight_init
import torch.nn.functional as F
from torch import distributions as pyd


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return


class DiagGaussianActor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, log_std_bounds):
        super().__init__()

        self.log_std_bounds = log_std_bounds
        self.trunk = mlp(obs_dim, hidden_dim, 2 * action_dim, hidden_depth)

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs):
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

        std = log_std.exp()

        self.outputs["mu"] = mu
        self.outputs["std"] = std

        dist = SquashedNormal(mu, std)
        return dist


import os
import glob
import importlib
from utils.logger_utils import get_logger
from eval.rollout import init_state_action_masks


class LBTransformerActor(nn.Module):
    def __init__(self, model_target, save_dir, exp_name):
        super().__init__()

        # initialize model
        ckpt_dir = os.path.join(save_dir, exp_name)
        ckpts = glob.glob(ckpt_dir + "/*.ckpt")
        ckpt_path = sorted(ckpts)[-1]

        logger = get_logger("actor")
        print("here")
        logger.info(f"loading model from: {ckpt_path}")

        model_cls = importlib.import_module(model_target).Model

        self.model = model_cls.load_from_checkpoint(
            checkpoint_path=ckpt_path, training=False, strict=False
        )
        self.model = self.model.cuda()

        (
            self.states,
            self.actions,
            self.timesteps,
            self.masks,
        ) = init_state_action_masks(
            self.model.state_dim, self.model.action_dim, 0, "cuda"
        )

    def reset(self):
        (
            self.states,
            self.actions,
            self.timesteps,
            self.masks,
        ) = init_state_action_masks(
            self.model.state_dim, self.model.action_dim, 0, "cuda"
        )

    def forward(self, obs):
        self.states = torch.cat([self.states, obs], dim=0)
        self.actions = torch.cat(
            [self.actions, torch.zeros((1, self.model.action_dim), device="cuda")],
            dim=0,
        )
        if self.timesteps.shape[-1] == 0:
            timestep = 0
        else:
            timestep = self.timesteps[:, -1] + 1

        self.timesteps = torch.cat(
            [
                self.timesteps,
                torch.ones((1, 1), device="cuda", dtype=torch.long) * timestep,
            ],
            dim=1,
        )

        action_preds, aux_pred, self.masks, _ = self.model.get_action(
            self.states,
            self.actions,
            self.timesteps,
            next_prompt="",
            predict_lang=True,
            use_model_generate=True,
            **self.masks,
        )
        # import ipdb

        # ipdb.set_trace()

        # dist = pyd.Normal(action_preds, scale=torch.zeros(9,).to("cuda"))

        # return dist
        return action_preds

    def forward_with_context(self, obs, context):
        action_preds, aux_pred, self.masks, _ = self.model.get_action(
            context["state"],
            self.actions,
            self.timesteps,
            next_prompt="",
            predict_lang=True,
            use_model_generate=True,
            **self.masks,
        )
