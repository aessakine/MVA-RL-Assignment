from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from segment_tree1 import MinSegmentTree, SumSegmentTree
import math
import os
import random
from collections import deque
from typing import Deque, Dict, List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output
from torch.nn.utils import clip_grad_norm_


env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.data = []
        self.index = 0
        self.device = 'cuda'
        self.size = 0
    
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)  # Add a placeholder if the buffer isn't full
        self.data[self.index] = (s, a, r, s_, d)  # Replace at circular index
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)  # Update size

    
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x: torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    
    def sample_batch_index(self, indices):
        batch = [self.data[idx] for idx in indices]  # Retrieve elements for each index
        return list(map(lambda x: torch.Tensor(np.array(x)).to(self.device), zip(*batch)))

    
    def get_n_step_info(self,n_step_buffer,gamma) :
        rew, next_obs, done = n_step_buffer[-1][-3:]
        for transition in reversed(list(n_step_buffer)[:-1]):
            r, n_o, d = transition[-3:]

            rew = r + gamma * rew * (1 - d)
            next_obs, done = (n_o, d) if d else (next_obs, done)

        return rew, next_obs, done
    def __len__(self) -> int:
        return self.size



class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, batch_size, alpha=0.6, device='cuda'):
        super().__init__(capacity)
        self.batch_size = batch_size
        self.alpha = alpha
        self.device = device
        self.max_priority, self.tree_ptr = 1.0, 0
        tree_capacity = 1
        while tree_capacity < capacity:
            tree_capacity *= 2
        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def append(self, s, a, r, s_, d):
        super().append(s, a, r, s_, d)
        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_batch(self, beta):
        assert len(self) >= self.batch_size
        assert beta > 0
        
        indices = self._sample_proportional()
        s, a, r, s_, d = self.sample_batch_index(indices)
        weights = torch.tensor([self._calculate_weight(i, beta) for i in indices]).to(self.device)
        return s, a, r, s_, d, weights, indices

    def update_priorities(self, indices, priorities):
        assert len(indices) == len(priorities)
        for idx, priority in zip(indices, priorities):
            assert priority > 0
            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha
            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self):
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
        return indices

    def _calculate_weight(self, idx, beta):
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        return weight / max_weight
    
def prefill_buffer(env, replay_buffer, prefill_steps):
    """Prefill the replay buffer with random transitions."""
    state, _ = env.reset()
    for _ in range(prefill_steps):
        action = env.action_space.sample()  # Take a random action
        next_state, reward, done, _, _ = env.step(action)
        replay_buffer.append(state, action, reward, next_state, done)
        
        if done:
            state, _ = env.reset()
        else:
            state = next_state


class NoisyLinear(nn.Module):
    def __init__(self,in_features,out_features,std_init=0.5) :
        super(NoisyLinear,self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        self.weight_mu = nn.Parameter(torch.Tensor(out_features,in_features)).to('cuda')
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features,in_features)).to('cuda')
        self.register_buffer('weight_epsilon',torch.Tensor(out_features,in_features).to('cuda'))
        
        self.bias_mu = nn.Parameter(torch.Tensor(out_features)).to('cuda')
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features)).to('cuda')
        self.register_buffer('bias_epsilon',torch.Tensor(out_features).to('cuda'))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self) :
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )
    
    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.
        
        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )
    
    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = torch.randn(size)

        return x.sign().mul(x.abs().sqrt())

class Network(nn.Module) :
    def __init__(self,in_dim,out_dim,atom_size,support) :
        super(Network,self).__init__()
        
        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size
        
        self.feature_layer = nn.Sequential(
            nn.Linear(in_dim, 128), 
            nn.ReLU(),
        )
        
        self.advantage_hidden_layer = NoisyLinear(128, 128)
        self.advantage_layer = NoisyLinear(128, out_dim * atom_size)

        # set value layer
        self.value_hidden_layer = NoisyLinear(128, 128)
        self.value_layer = NoisyLinear(128, 1)
    
    def forward(self,x) :    
        dist = self.dist(x)
        q = torch.sum(dist*self.support,dim=2)
        return q
    
    def dist(self,x) :
        feature = self.feature_layer(x)
        avd_hid = F.relu(self.advantage_hidden_layer(feature))
        val_hid = F.relu(self.value_hidden_layer(feature))
        
        advantage = self.advantage_layer(avd_hid).view(-1,self.out_dim,self.atom_size)
        q_atoms = self.value_layer(val_hid).view(-1,1,self.atom_size)
        dist = q_atoms + advantage - advantage.mean(dim=1, keepdim=True)
        dist = F.softmax(dist,d)
        print(dist)
        dist = dist.clamp(min=1e-3)
        
        return dist
    
    def reset_noise(self):
        """Reset all noisy layers."""
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()
                
class ProjectAgent:
    def __init__(self,env) :
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        self.env = env
        self.batch_size = 256
        self.target_update = 20
        self.gamma = 0.99
        self.alpha = 0.2
        self.memory_size = 2 ** 17
        self.n_step = 1
        
        self.beta = 0.6
        self.prior_eps = 1e-6
        self.memory = PrioritizedReplayBuffer(self.memory_size,batch_size=256)
        
        self.use_n_step = True if self.n_step > 1 else False
        if self.use_n_step :
            self.memory_n = ReplayBuffer(self.memory_size)  
        
        self.v_min = 0.0
        self.v_max = 200.0
        self.atom_size = 256
        self.support = torch.linspace(
            self.v_min, self.v_max, self.atom_size
        ).to('cuda')

        self.dqn = Network(
            state_dim, action_dim, self.atom_size, self.support
        ).to('cuda')
        self.dqn_target = Network(
            state_dim, action_dim, self.atom_size, self.support
        ).to('cuda')
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()
        
        self.optimizer = optim.Adam(self.dqn.parameters())

        self.transition = list()       
        
    def select_action(self,state) :
        selected_action = self.dqn(torch.FloatTensor(state).to('cuda')).argmax()
        
        return selected_action
    
    def step(self,action,state) :
        next_state, reward, done, trunc, _ = env.step(action)
        self.transition += [state,action,reward,next_state,done or trunc] 
        if self.use_n_step :
            one_step_transition = self.memory_n.append(*self.transition)
        else :
            one_step_transition = self.transition
        if self.use_n_step :
            self.memory_n.append(*one_step_transition)
        return next_state,reward,done
    
    def update_model(self): 
        s, a, r, s_, d,weights,indices  = self.memory.sample_batch(self.beta)
        elementwise_loss = self._compute_dqn_loss(s, a, r, s_, d,weights,indices,self.gamma) 
        loss = torch.mean(elementwise_loss*weights)
        if self.use_n_step :
            gamma = self.gamma ** self.n_step
            s, a, r, s_, d,weights,indices = self.memory_n.sample_batch_index(indices)
            elementwise_loss_n_loss = self._compute_dqn_loss(s, a, r, s_, d,weights,indices, gamma)
            elementwise_loss += elementwise_loss_n_loss   
            loss = torch.mean(elementwise_loss*weights)
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(),10.0)
        self.optimizer.step()
        
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)
        
        # NoisyNet: reset noise
        self.dqn.reset_noise()
        self.dqn_target.reset_noise()

        return loss.item()
    
    def train(self,max_episode,plotting_interval = 200) :
        state,_ = self.env.reset()
        update_cnt = 0
        losses = []
        scores = []
        score = 0
        step = 0
        episode = 0
        loss = 5.
        
        while episode < max_episode: 
            action = self.select_action(state)
            next_state,reward,done = self.step(action,state)
            
            state = next_state
            score += reward
            
            fraction = min(episode/max_episode,1.0)
            self.beta = self.beta + fraction * (1.0 - self.beta)
            
            if done or step % 200 == 0:
                state,_ = self.env.reset()
                scores.append(score)
                print("Episode ",
                    "{:3d}".format(episode),
                    ', step ',', score ',
                    "{:.2e}".format(score),
                    " , loss ",
                    loss == 0,
                    sep="",)
                score = 0
                episode +=1
            
            loss = self.update_model()
            losses.append(loss)
            update_cnt += 1
            step +=1
            
            if update_cnt % self.target_update == 0:
                self._target_hard_update()
            


    def act(self, observation, use_random=False):
        state,_ = env.reset()
        done = False
        score = 0
        while not done :
            action = self.select_action(state)
            next_state,reward,done = self.step(action)
            
            state = next_state
            score += reward
        return score

    def _compute_dqn_loss(self,s, a, r, s_, d,weights,indices, gamma) :
        delta_z = float(self.v_max - self.v_min)/(self.atom_size - 1)
        
        with torch.no_grad():
            next_action = self.dqn(s_).argmax(1)
            next_dist = self.dqn_target.dist(s_)
            next_dist = next_dist[range(self.batch_size),next_action]
            
            t_z = r + (1 - d)* gamma * self.support
            t_z = t_z.clamp(min=self.v_min,max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()
            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                ).long()
                .unsqueeze(1)
                .expand(self.batch_size, self.atom_size).
                to('cuda'))
            proj_dist = torch.zeros(next_dist.size(), device='cuda')
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.dqn.dist(s).squeeze(1)
        a = a.long()
        log_p = torch.log(dist[range(self.batch_size), a])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss
    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())
                
    def _plot(
        self, 
        frame_idx: int, 
        scores: List[float], 
        losses: List[float],
    ):
        """Plot the training progresses."""
        clear_output(True)
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('frame %s. score: %s' % (frame_idx, np.mean(scores[-10:])))
        plt.plot(scores)
        plt.subplot(132)
        plt.title('loss')
        plt.plot(losses)
        plt.show()
    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, weights_only=True))
        self.target_model.load_state_dict(self.model.state_dict())
        
if __name__ == "__main__":
    state_dim = env.observation_space.shape[0]
    env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
) 
    n_action = env.action_space.n 
    nb_neurons=256
    agent = ProjectAgent(env)
    prefill_steps = int(agent.memory_size * 0.1)  # Fill 10% of the buffer
    prefill_steps = 512
    print(f"Prefilling the replay buffer with {prefill_steps} transitions...")
    prefill_buffer(env, agent.memory, prefill_steps)
    
    agent.train(max_episode=200)
