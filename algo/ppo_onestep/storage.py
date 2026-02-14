import torch
from torch.utils.data.sampler import BatchSampler, SequentialSampler, SubsetRandomSampler


class RolloutStorage:
    def __init__(self, num_envs, num_transitions_per_env, obs_shape, states_shape, actions_shape, device="cpu", sampler="sequential"):
        self.device = device
        self.sampler = sampler

        # Core
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        self.states = torch.zeros(num_transitions_per_env, num_envs, *states_shape, device=self.device)
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # For PPO
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        self.step = 0
    
    def change_num_envs(self, new_num_envs):
        assert self.step == 0, "Can only change num_envs at the beginning of rollout"
        self.num_envs = new_num_envs
        self.observations = torch.zeros(self.num_transitions_per_env, self.num_envs, *self.observations.shape[2:], device=self.device)
        self.states = torch.zeros(self.num_transitions_per_env, self.num_envs, *self.states.shape[2:], device=self.device)
        self.rewards = torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=self.device)
        self.actions = torch.zeros(self.num_transitions_per_env, self.num_envs, *self.actions.shape[2:], device=self.device)
        self.dones = torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=self.device).byte()
        self.actions_log_prob = torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=self.device)
        self.values = torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=self.device)
        self.returns = torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=self.device)
        self.advantages = torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=self.device)
        self.mu = torch.zeros(self.num_transitions_per_env, self.num_envs, *self.actions.shape[2:], device=self.device)
        self.sigma = torch.zeros(self.num_transitions_per_env, self.num_envs, *self.actions.shape[2:], device=self.device)

    def add_transitions(self, observations, states, actions, rewards, dones, values, actions_log_prob, mu, sigma):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")

        self.observations[self.step].copy_(observations)
        self.states[self.step].copy_(states)
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards.view(-1, 1))
        self.dones[self.step].copy_(dones.view(-1, 1))
        self.values[self.step].copy_(values)
        self.actions_log_prob[self.step].copy_(actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(mu)
        self.sigma[self.step].copy_(sigma)

        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values=None, gamma=None, lam=None):
        self.returns = self.rewards.clone()
        # advantages -> r - V(s)
        self.advantages = self.rewards - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_statistics(self):
        done = self.dones.cpu()
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = done_indices[1:] - done_indices[:-1]
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

        if self.sampler == "sequential":
            # For physics-based RL, each environment is already randomized. There is no value to doing random sampling
            # but a lot of CPU overhead during the PPO process. So, we can just switch to a sequential sampler instead
            subset = SequentialSampler(range(batch_size))
        elif self.sampler == "random":
            subset = SubsetRandomSampler(range(batch_size))

        batch = BatchSampler(subset, mini_batch_size, drop_last=True)
        return batch
