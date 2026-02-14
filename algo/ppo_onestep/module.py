import numpy as np

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from typing import Optional

from ..pn_utils.maniskill_learn.networks.backbones.pointnet import getPointNet
from ..pn_utils.maniskill_learn.networks.backbones.pointnet import getPointNetWithInstanceInfo
from isaacgymenvs.utils.torch_jit_utils import *


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError("invalid activation function!")


class PointNetBackbone(nn.Module):
    def __init__(self, pc_dim: int, feature_dim: int, pretrained_model_path: Optional[str] = None):
        super().__init__()
        # self.save_hyperparameters()
        self.pc_dim = pc_dim
        self.feature_dim = feature_dim
        self.backbone = getPointNet(
            {"input_feature_dim": self.pc_dim, "feat_dim": self.feature_dim})

        if pretrained_model_path is not None:
            print("Loading pretrained model from:", pretrained_model_path)
            state_dict = torch.load(pretrained_model_path, map_location="cpu")[
                "state_dict"]
            missing_keys, unexpected_keys = self.load_state_dict(
                state_dict, strict=False)
            if len(missing_keys) > 0:
                print("missing_keys:", missing_keys)
            if len(unexpected_keys) > 0:
                print("unexpected_keys:", unexpected_keys)

    def forward(self, input_pc):
        return self.backbone(input_pc)


class TransPointNetBackbone(nn.Module):
    def __init__(self, pc_dim: int = 6, feature_dim: int = 128, state_dim: int = 191 + 29, use_seg: bool = True):
        super().__init__()

        cfg = {}
        cfg["state_dim"] = state_dim
        cfg["feature_dim"] = feature_dim
        cfg["pc_dim"] = pc_dim
        cfg["output_dim"] = feature_dim
        if use_seg:
            cfg["mask_dim"] = 2
        else:
            cfg["mask_dim"] = 0

        self.transpn = getPointNetWithInstanceInfo(cfg)

    def forward(self, input_pc):
        input_pc["pc"] = torch.cat([input_pc["pc"], input_pc["mask"]], dim=-1)
        return self.transpn(input_pc)


from torch.distributions import Normal, Independent

def atanh(x, eps=1e-6):
    # clamp to avoid NaNs at |x|=1
    x = torch.clamp(x, -1 + eps, 1 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))

def tanh_squash_and_log_prob(dist_base: Independent, pre_tanh: torch.Tensor, eps: float = 1e-6):
    """
    Given a base (unsquashed) diagonal Gaussian dist and samples pre_tanh ~ N(mu, sigma),
    return squashed actions a = tanh(pre_tanh) and corrected log_prob(a).
    """
    # Base log prob of pre-tanh sample
    log_prob_pre = dist_base.log_prob(pre_tanh)
    # Change-of-variables correction: sum log(1 - tanh(x)^2)
    # Use a numerically stable form: log(1 - tanh(x)^2) = 2*(log(2) - x - softplus(-2x))
    # but the simple version below with clamp is fine for PPO:
    a = torch.tanh(pre_tanh)
    log_det_jacob = torch.sum(torch.log(1 - a.pow(2) + eps), dim=-1)
    log_prob = log_prob_pre - log_det_jacob
    return a, log_prob

class ActorCritic(nn.Module):
    def __init__(self, obs_shape, states_shape, actions_shape, initial_std, model_cfg, 
                 asymmetric=False, use_pcl=False):
        super().__init__()
        self.asymmetric = asymmetric
        self.use_pcl = use_pcl
        self.backbone_type = model_cfg['backbone_type']
        self.freeze_backbone = model_cfg["freeze_backbone"]

        if model_cfg is None:
            actor_hidden_dim = [256, 256, 256]
            critic_hidden_dim = [256, 256, 256]
            activation = nn.SELU()
        else:
            actor_hidden_dim = model_cfg["pi_hid_sizes"]
            critic_hidden_dim = model_cfg["vf_hid_sizes"]
            activation = get_activation(model_cfg["activation"])
        
        if self.use_pcl:
            self.pc_shape = model_cfg['pc_shape'] # [512,3]
            self.pc_emb_dim = model_cfg["pc_emb_dim"]
            if self.backbone_type == "pn":
                self.backbone = PointNetBackbone(pc_dim=self.pc_shape[-1], feature_dim=self.pc_emb_dim)
            else:
                raise ValueError(f"Invalid backbone type: {self.backbone_type}")
            #print(self.backbone)
        else:
            self.backbone = None
            self.pc_emb_dim = 0
            self.pc_shape = [0,0]

        self.num_obs = obs_shape[0]
        self.num_state_based_obs = self.num_obs - np.prod(self.pc_shape) + self.pc_emb_dim # replace N*3 pc with pn embedding
        self.pc_start_idx = self.num_obs - np.prod(self.pc_shape)
        self.act_dim = actions_shape[0] if isinstance(actions_shape, (list, tuple)) else int(actions_shape)

        # Actor
        actor_layers = [nn.Linear(self.num_state_based_obs, actor_hidden_dim[0]), activation]
        for l in range(len(actor_hidden_dim)):
            if l == len(actor_hidden_dim) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], self.act_dim))
            else:
                actor_layers += [nn.Linear(actor_hidden_dim[l], actor_hidden_dim[l + 1]), activation]
        self.actor_mean = nn.Sequential(*actor_layers)

        # Critic
        critic_layers = [nn.Linear(self.num_state_based_obs, critic_hidden_dim[0]), activation]
        for l in range(len(critic_hidden_dim)):
            if l == len(critic_hidden_dim) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], 1))
            else:
                critic_layers += [nn.Linear(critic_hidden_dim[l], critic_hidden_dim[l + 1]), activation]
        self.critic = nn.Sequential(*critic_layers)

        # Log-std parameter (diagonal)
        init_log_std = float(np.log(initial_std))
        self.log_std = nn.Parameter(torch.full((self.act_dim,), init_log_std))

        # Initialize the weights like in Stable Baselines
        self.init_orthogonal_(self.actor_mean, [np.sqrt(2)] * len(actor_hidden_dim) + [0.01])
        self.init_orthogonal_(self.critic,     [np.sqrt(2)] * len(critic_hidden_dim) + [1.0])

    @staticmethod
    def init_orthogonal_(sequential, gains):
        idx = 0
        for m in sequential:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=gains[idx])
                nn.init.zeros_(m.bias)
                idx += 1

    def forward(self, observations, states=None, inference=False):
        """
        Returns (actions, actions_log_prob, value, actions_mean_squashed, log_std_vector)
        - actions in [-1, 1] via tanh-squash
        """
        if self.use_pcl and not self.freeze_backbone:
            if self.backbone_type =="pn":
                pc = observations[:, self.pc_start_idx:].reshape(-1, *self.pc_shape)
                pc_feature = self.backbone(pc).reshape(-1, self.pc_emb_dim)
            else:
                raise NotImplementedError
            observations = torch.cat([observations[:, :self.pc_start_idx], pc_feature], dim=1)
            mean = self.actor_mean(observations)
        elif self.use_pcl and self.freeze_backbone:
            with torch.no_grad():
                raise NotImplementedError
        else:
            mean = self.actor_mean(observations) # pre-tanh mean

        std = self.log_std.exp()
        base = Independent(Normal(mean, std), 1)

        if inference:
            # Deterministic (mean) action, squashed to bounds
            actions = torch.tanh(mean)
            # if self.asymmetric:
            #     value = self.critic(states)
            # else:
            #     value = self.critic(observations)
            return actions.detach()

        # Reparameterized sample: mean + std * eps ; we can call .rsample() for gradients if needed
        pre_tanh = base.rsample()
        actions, log_prob = tanh_squash_and_log_prob(base, pre_tanh)

        # Critic
        value = self.critic(states) if self.asymmetric else self.critic(observations)

        return (
            actions.detach(),
            log_prob.detach(),
            value.detach(),
            torch.tanh(mean).detach(),                # squashed mean for logging
            self.log_std.expand(mean.shape[0], -1).detach(),
        )

    def evaluate(self, observations, states, actions, eps: float = 1e-6):
        """
        Evaluate log_prob/entropy/value at given (already squashed) actions.
        """
        if self.use_pcl and not self.freeze_backbone:
            if self.backbone_type =="pn":
                pc = observations[:, self.pc_start_idx:].reshape(-1, *self.pc_shape)
                pc_feature = self.backbone(pc).reshape(-1, self.pc_emb_dim)
            else:
                raise NotImplementedError
            observations = torch.cat([observations[:, :self.pc_start_idx], pc_feature], dim=1)
            mean = self.actor_mean(observations)
        elif self.use_pcl and self.freeze_backbone:
            with torch.no_grad():
                raise NotImplementedError
        else:
            mean = self.actor_mean(observations) # pre-tanh mean
            
        std = self.log_std.exp()
        base = Independent(Normal(mean, std), 1)

        # Map actions in [-1,1] back to pre-tanh space
        pre_tanh = atanh(actions, eps=eps)

        # Corrected log prob under the squashed policy
        log_prob_pre = base.log_prob(pre_tanh)
        log_det_jacob = torch.sum(torch.log(1 - actions.pow(2) + eps), dim=-1)
        log_prob = log_prob_pre - log_det_jacob

        # Entropy: use base Gaussian entropy (standard practice in SAC / PPO with squashing)
        entropy = base.entropy()

        value = self.critic(states) if self.asymmetric else self.critic(observations)

        return (
            log_prob,
            entropy,
            value,
            torch.tanh(mean),                          # squashed mean
            self.log_std.expand(mean.shape[0], -1),
        )