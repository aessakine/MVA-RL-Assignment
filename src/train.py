from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import random
import numpy as np
import torch
from evaluate import evaluate_HIV, evaluate_HIV_population
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F


env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)
env1 = TimeLimit(
    env=HIVPatient(domain_randomization=True), max_episode_steps=200
)
# The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.
# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
state_dim = env.observation_space.shape[0]
n_action = env.action_space.n 
    
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = int(capacity) # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)


class DuelingDQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)  # Increased neurons
        self.fc2 = nn.Linear(1024, 512)       # Added an additional layer with more neurons
        self.fc3 = nn.Linear(512, 256)
        self.fc_value = nn.Linear(256, 128)
        self.value = nn.Linear(128, 1)
        self.fc_advantage = nn.Linear(256, 128)
        self.advantage = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.silu(self.fc1(x))  # Swish activation (SiLU)
        x = F.silu(self.fc2(x))
        x = F.silu(self.fc3(x))
        value = F.silu(self.fc_value(x))
        value = self.value(value)
        advantage = F.silu(self.fc_advantage(x))
        advantage = self.advantage(advantage)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


def action_greedy(epsilon,env,state,model) :
    if np.random.rand() < epsilon:
        action = env.action_space.sample()
    else:
        with torch.no_grad():
            Q = model(torch.Tensor(state).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu'))
            action = torch.argmax(Q).item()  
    return action  

def prefill_replay_buffer(env, replay_buffer, prefill_steps=1000):
    print(f"Prefilling the replay buffer with {prefill_steps} transitions...")
    state, _ = env.reset()
    for _ in range(prefill_steps):
        action = env.action_space.sample()  # Random action
        next_state, reward, done, _, _ = env.step(action)
        replay_buffer.append(state, action, reward, next_state, done)
        
        if done:
            state, _ = env.reset()
        else:
            state = next_state
    print(f"Replay buffer prefilled with {len(replay_buffer)} transitions.")
    

DQN = DuelingDQNNetwork(state_dim,n_action).to('cuda' if torch.cuda.is_available() else 'cpu')

class ProjectAgent:
    def __init__(self, env=env, model=DQN):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.nb_actions = env.action_space.n
        self.gamma = 0.99
        self.batch_size = 256
        buffer_size = int(1e6)
        self.memory = ReplayBuffer(buffer_size)
        self.epsilon_max = 0.3
        self.epsilon_min = 0.01
        self.period = 7500  # Period for epsilon oscillation (steps)
        self.model = model
        self.target_model = deepcopy(self.model).to(device)
        self.criterion = torch.nn.SmoothL1Loss()
        lr = 1e-3
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr)
        self.nb_gradient_steps = 50
        self.update_target_freq = 20
        self.update_target_tau = 0.005
        self.device = "cuda"
        self.best_return = -float("inf")
        self.best_return1 = -float('inf')
        self.update_target_strategy = 'ema'

    def act(self, observation, use_random=False):

        state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            # Double DQN
            with torch.no_grad():
                # Action selection using the policy network
                next_actions = self.model(Y).argmax(1)
                # Action evaluation using the target network
                QYmax = self.target_model(Y).gather(1, next_actions.unsqueeze(1)).squeeze(1)

            # Compute the target update
            update = torch.addcmul(R, 1 - D, QYmax, value=self.gamma)

            # Current Q values
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1)).squeeze(1)

            # Compute the loss
            loss = self.criterion(QXA, update)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self, save_path="best_hiv_agent.pth", max_episode=200):
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        state1, _ = env1.reset()
        step = 0
        prefill_replay_buffer(env, self.memory, prefill_steps=1000)
        prefill_replay_buffer(env1, self.memory, 1000)
        epsilon = 0.05

        while episode < max_episode:
            # Strictly periodic epsilon decay
            #epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * (
            #    0.5 + 0.5 * np.sin(2 * np.pi * step / self.period)
            #)

            action = action_greedy(epsilon, env, state, self.model)
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            action1 = action_greedy(epsilon, env1, state1, self.model)
            next_state1, reward1, done1, trunc1, _ = env1.step(action1)
            self.memory.append(state1, action1, reward1, next_state1, done1)
            episode_cum_reward += reward 

            for _ in range(self.nb_gradient_steps):
                self.gradient_step()

            if self.update_target_strategy == 'replace':
                if step % self.update_target_freq == 0: 
                    self.target_model.load_state_dict(self.model.state_dict())
            if self.update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau * model_state_dict[key] + (1 - tau) * target_state_dict[key]
                self.target_model.load_state_dict(target_state_dict)

            step += 1
            if done or trunc:
                episode += 1
                score_agent: float = evaluate_HIV(agent=agent, nb_episode=1)
                print(
                    "Episode ",
                    "{:3d}".format(episode),
                    "epsilon ",
                    "{:.2e}".format(epsilon),
                    ', step ',
                    step,
                    ", batch size ",
                    "{:5d}".format(len(self.memory)),
                    ", episode return ",
                    "{:.2e}".format(episode_cum_reward),
                    ', score ',
                    "{:.2e}".format(score_agent),
                    sep="",
                )
                if episode_cum_reward > self.best_return:
                    self.best_return = episode_cum_reward
                    self.save(save_path)
                if score_agent > self.best_return1:
                    self.best_return1 = score_agent
                    self.save('best_score.pth')
                    print('The model was saved successfully with score ', self.best_return1)
                state, _ = env.reset()
                state1, _ = env1.reset()
                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0
            else:
                state = next_state
                state1 = next_state1

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        torch.save(self.target_model.state_dict(),'best_hiv_target.pth')

    def load(self, path):
        self.model.load_state_dict(torch.load(path, weights_only=True,map_location=torch.device('cpu')))
        self.target_model.load_state_dict(self.model.state_dict())



if __name__ == "__main__":
    state_dim = env.observation_space.shape[0]
    n_action = env.action_space.n 
    nb_neurons=256

    
    agent = ProjectAgent(env,DQN)
    agent.train()
