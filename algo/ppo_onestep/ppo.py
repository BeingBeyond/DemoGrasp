import os
import time
from gym.spaces import Space

import statistics
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import numpy as np

from .storage import RolloutStorage

from tqdm import tqdm

from isaacgymenvs.utils.torch_jit_utils import *
import sys
import os

# Add the parent directory to the Python path for imports
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
# if parent_dir not in sys.path:
#     sys.path.append(parent_dir)


class PPO:
    def __init__(self, vec_env, actor_critic_class, train_param, log_dir="run", apply_reset=False, action_dim=6):
        # other parameters
        self.is_testing_all_objects = train_param.get("is_testing_all_objects", False)
        self.times_testing_all_objects = train_param.get("times_testing_all_objects", 200)
        self.plan = train_param.get("plan", False)

        # PPO parameters
        self.clip_param = train_param["cliprange"]
        self.num_learning_epochs = train_param["noptepochs"]
        self.num_mini_batches = train_param["nminibatches"]
        self.num_learning_iterations = train_param["max_iterations"]
        self.num_transitions_per_env = train_param["nsteps"]
        self.value_loss_coef = train_param.get("value_loss_coef", 2.0)
        self.entropy_coef = train_param["ent_coef"]
        self.gamma = train_param["gamma"]
        self.lam = train_param["lam"]
        self.max_grad_norm = train_param.get("max_grad_norm", 2.0)
        self.use_clipped_value_loss = train_param.get("use_clipped_value_loss", False)
        self.init_noise_std = train_param.get("init_noise_std", 0.3)
        self.discard_invalid_resets = train_param["discard_invalid_resets"]

        self.model_cfg = train_param.policy
        self.sampler = train_param.get("sampler", "sequential")
        self.is_vision = train_param.is_vision

        if not isinstance(vec_env.observation_space, Space) or not isinstance(vec_env.state_space, Space) or not isinstance(vec_env.action_space, Space):
            raise TypeError("vec_env.observation_space, vec_env.state_space and vec_env.action_space must be gym Spaces")

        self.observation_space = vec_env.observation_space
        #self.action_space = vec_env.action_space
        self.state_space = vec_env.state_space
        self.device = vec_env.device

        # for DemoGrasp training
        self.action_space = torch.zeros(action_dim, device=self.device)
        self.asymmetric = vec_env.num_states > 0

        self.desired_kl = train_param.get("desired_kl", None)
        self.schedule = train_param.get("schedule", "fixed")
        self.step_size = train_param["optim_stepsize"]

        # PPO components
        self.vec_env = vec_env
        self.actor_critic = actor_critic_class(
            self.observation_space.shape,
            self.state_space.shape,
            self.action_space.shape,
            self.init_noise_std,
            self.model_cfg,
            asymmetric=self.asymmetric,
            use_pcl=self.is_vision,
        )
        self.actor_critic.to(self.device)
        self.storage = RolloutStorage(
            self.vec_env.num_envs,
            self.num_transitions_per_env,
            self.observation_space.shape,
            self.state_space.shape,
            self.action_space.shape,
            self.device,
            self.sampler,
        )
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.step_size)
        print(self.actor_critic)

        # Log
        self.save_interval = train_param["save_interval"]
        self.log_dir = log_dir
        self.print_log = train_param["print_log"]
        self.tot_timesteps = 0
        self.tot_time = 0
        self.is_testing = train_param["test"]
        self.current_learning_iteration = 0
        if not self.is_testing:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        self.apply_reset = apply_reset

    def load(self, path, is_testing=False):
        self.actor_critic.load_state_dict(torch.load(path, map_location=self.device))
        if is_testing:
            self.actor_critic.eval()
        else:
            # # extract the xxx/model_iteration_num.pt
            # path = path.split("/")[-1]
            # self.current_learning_iteration = int(path.split("_")[-1].split(".")[0])
            self.current_learning_iteration = 0
            self.actor_critic.train()

    def save(self, path):
        torch.save(self.actor_critic.state_dict(), path)

    def test(self, ckpt_path):
        self.load(ckpt_path, is_testing=True)

    def run(self):
        num_learning_iterations = self.num_learning_iterations
        if self.is_testing:
            for round in range(10):
                per_env_rgb = [[] for _ in range(self.vec_env.num_envs)]
                current_obs = self.vec_env.reset_idx(torch.arange(self.vec_env.num_envs))["obs"]
                current_states = self.vec_env.get_state()
                #actions, actions_log_prob, values, mu, sigma= self.actor_critic(current_obs, current_states) #(num_envs, 6)
                actions = self.actor_critic(current_obs, current_states, inference=True)
                #print(actions)
                self.vec_env.generate_reaching_plan_idx(torch.arange(self.vec_env.num_envs), actions=actions)
                #dones = torch.ones(self.vec_env.num_envs, device=self.device) # to make sure env reset after each step
                for t in range(self.vec_env.max_episode_length):
                    env_action = self.vec_env.compute_reference_actions()
                    obs, reward, reset, extras = self.vec_env.step(env_action)
                    if (t == self.vec_env.max_episode_length - 2):
                        successs  = self.vec_env.successes.clone().to(self.device)
                        print(f"Round {round}, success rate: {successs.sum()/successs.shape[0]}")
            exit(0)

        else:
            rewbuffer = deque(maxlen=self.vec_env.num_envs) #100)
            lenbuffer = deque(maxlen=self.vec_env.num_envs) #100)
            cur_reward_sum = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            cur_episode_length = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)

            reward_sum = []
            episode_length = []

            for it in range(self.current_learning_iteration, num_learning_iterations):
                start = time.time()
                ep_infos = []

                # Rollout
                if self.discard_invalid_resets:
                    assert self.num_transitions_per_env == 1, "Currently only support discard_invalid_resets with nsteps=1"
                for _ in range(self.num_transitions_per_env):
                    # Compute the action
                    current_obs = self.vec_env.reset_idx(torch.arange(self.vec_env.num_envs))["obs"]
                    current_states = self.vec_env.get_state()
                    if self.discard_invalid_resets:
                        is_init_state_valid = self.vec_env.is_init_state_valid.clone()
                        self.storage.change_num_envs(is_init_state_valid.sum().item())

                    actions, actions_log_prob, values, mu, sigma= self.actor_critic(current_obs, current_states) #(num_envs, 6)
                    self.vec_env.generate_reaching_plan_idx(torch.arange(self.vec_env.num_envs), actions=actions)
                    dones = torch.ones(self.vec_env.num_envs, device=self.device) # to make sure env reset after each step
                    for t in range(self.vec_env.max_episode_length):
                        env_action = self.vec_env.compute_reference_actions()
                        obs, reward, reset, extras = self.vec_env.step(env_action)
                        if (t == self.vec_env.max_episode_length - 2):
                            rews  = self.vec_env.successes.clone().to(self.device)
                            rews = torch.where(
                                self.vec_env.has_hit_table,
                                torch.zeros_like(rews),
                                rews
                            ) # if keypoints lower than tabletop, set reward to 0
                            break
                    # Record the transition
                    if not self.discard_invalid_resets:
                        self.storage.add_transitions(current_obs, current_states, actions, rews, dones, values, actions_log_prob, mu, sigma)
                    else:
                        #print(is_init_state_valid)
                        self.storage.add_transitions(
                            current_obs[is_init_state_valid], 
                            current_states[is_init_state_valid], 
                            actions[is_init_state_valid], 
                            rews[is_init_state_valid], 
                            dones[is_init_state_valid], 
                            values[is_init_state_valid], 
                            actions_log_prob[is_init_state_valid], 
                            mu[is_init_state_valid], 
                            sigma[is_init_state_valid]
                        )

                    if self.print_log:
                        cur_reward_sum[:] += rews
                        cur_episode_length[:] += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        reward_sum.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        episode_length.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                if self.print_log:
                    rewbuffer.extend(reward_sum)
                    lenbuffer.extend(episode_length)


                stop = time.time()
                collection_time = stop - start

                mean_trajectory_length, mean_reward = self.storage.get_statistics()
                current_success_rate = self.vec_env.successes.mean().item()
                current_hit_table_rate = self.vec_env.has_hit_table.float().mean().item()
                if self.discard_invalid_resets:
                    current_success_rate_valid_envs = self.vec_env.successes[self.vec_env.is_init_state_valid].mean().item()
                    current_hit_table_rate_valid_envs = self.vec_env.has_hit_table[self.vec_env.is_init_state_valid].float().mean().item()
                    current_reset_valid_rate = self.vec_env.is_init_state_valid.float().mean().item()

                # Learning step
                start = time.time()
                self.storage.compute_returns(None, self.gamma, self.lam)
                mean_value_loss, mean_surrogate_loss = self.update()
                self.storage.clear()
                stop = time.time()
                learn_time = stop - start

                if self.print_log:
                    self.log(locals())

                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, "model_{}.pt".format(it)))

                ep_infos.clear()

            self.save(os.path.join(self.log_dir, "model_{}.pt".format(num_learning_iterations)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_transitions_per_env * self.vec_env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = f""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    try:
                        infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                    except:
                        breakpoint()
                value = torch.mean(infotensor)
                self.writer.add_scalar("Episode/" + key, value, locs["it"])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.actor_critic.log_std.exp().mean()

        self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
        self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
            self.writer.add_scalar("Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time)

        self.writer.add_scalar("Train/mean_reward/step", locs["mean_reward"], locs["it"])
        self.writer.add_scalar("Train/mean_episode_length/episode", locs["mean_trajectory_length"], locs["it"])
        self.writer.add_scalar("Train/current_success_rate", locs["current_success_rate"], locs["it"])
        self.writer.add_scalar("Train/current_hit_table_rate", locs["current_hit_table_rate"], locs["it"])
        if self.discard_invalid_resets:
            self.writer.add_scalar("Train/current_success_rate_valid_envs", locs["current_success_rate_valid_envs"], locs["it"])
            self.writer.add_scalar("Train/current_hit_table_rate_valid_envs", locs["current_hit_table_rate_valid_envs"], locs["it"])
            self.writer.add_scalar("Train/current_reset_valid_rate", locs["current_reset_valid_rate"], locs["it"])

        fps = int(self.num_transitions_per_env * self.vec_env.num_envs / (locs["collection_time"] + locs["learn_time"]))

        str = f" \033[1m Learning iteration {locs['it']}/{locs['num_learning_iterations']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n"""
                f"""{'Current success rate:':>{pad}} {locs["current_success_rate"]:.2f}\n"""
                f"""{'Current hit table rate:':>{pad}} {locs["current_hit_table_rate"]:.2f}\n"""
            )
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n"""
                f"""{'Current success rate:':>{pad}} {locs["current_success_rate"]:.2f}\n"""
                f"""{'Current hit table rate:':>{pad}} {locs["current_hit_table_rate"]:.2f}\n"""
            )

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)

    def update(self):
        
        mean_value_loss = 0
        mean_surrogate_loss = 0

        batch = self.storage.mini_batch_generator(self.num_mini_batches)
        for epoch in range(self.num_learning_epochs):
            for indices in batch:
                # print(indices)
                obs_batch = self.storage.observations.view(-1, *self.storage.observations.size()[2:])[indices]
                if self.asymmetric:
                    states_batch = self.storage.states.view(-1, *self.storage.states.size()[2:])[indices]
                else:
                    states_batch = None
                
                actions_batch = self.storage.actions.view(-1, self.storage.actions.size(-1))[indices]
                target_values_batch = self.storage.values.view(-1, 1)[indices]
                returns_batch = self.storage.returns.view(-1, 1)[indices]
                old_actions_log_prob_batch = self.storage.actions_log_prob.view(-1, 1)[indices]
                advantages_batch = self.storage.advantages.view(-1, 1)[indices]
                old_mu_batch = self.storage.mu.view(-1, self.storage.actions.size(-1))[indices]
                old_sigma_batch = self.storage.sigma.view(-1, self.storage.actions.size(-1))[indices]

                (
                    actions_log_prob_batch,
                    entropy_batch,
                    value_batch,
                    mu_batch,
                    sigma_batch,
                ) = self.actor_critic.evaluate(obs_batch, states_batch, actions_batch)

                # KL
                if self.desired_kl != None and self.schedule == "adaptive":
                    kl = torch.sum(sigma_batch - old_sigma_batch + (torch.square(old_sigma_batch.exp()) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch.exp())) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.step_size = max(1e-5, self.step_size / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.step_size = min(1e-2, self.step_size * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.step_size

                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                # for name, param in self.actor_critic.actor.named_parameters():
                #     print(name, param.grad)

                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates

        return mean_value_loss, mean_surrogate_loss

