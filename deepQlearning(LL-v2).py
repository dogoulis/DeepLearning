# download this in order to load the game:
#!pip install box2d-py

# =====================================================================================================================

# IMPORTS
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time

# =====================================================================================================================

# generating the environment:
env = gym.make('LunarLander-v2')

# defining a simple function for time:

def done_time(startTime, endTime):
    t = endTime - startTime
    tmins = int(t / 60)
    tsecs = int(t - (tmins * 60))
    return tmins, tsecs

# ======================================================================================================================

# checking if CUDA is available:

is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

# ======================================================================================================================

# class for the fully connected deep Q network:


class DQN(nn.Module):

  def __init__(self, in_dims, fc1_dims, fc2_dims, fc3_dims, out_dims, learning_rate):
    super(DQN, self).__init__()
    self.in_dims = in_dims
    self.out_dims = out_dims
    self.fc1_dims = fc1_dims
    self.fc2_dims = fc2_dims
    self.fc3_dims = fc3_dims
    self.fc1 = nn.Linear(*self.in_dims, self.fc1_dims)
    self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
    self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
    self.fc4 = nn.Linear(self.fc3_dims, self.out_dims)
    self.optimizer = optim.Adam(self.parameters(), lr = learning_rate)
    self.loss = nn.MSELoss()
    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    self.to(self.device)


  def forward(self, state):
    x = F.relu(self.fc1(state))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    actions = self.fc4(x)

    return actions

# ======================================================================================================================

# class for the Agent:

class Agent():
    def __init__(self, gamma, epsilon, learning_rate, in_dims, batch_size,
                 numof_actions, max_mem_size = 100000, eps_min = 0.01,
                 eps_reduc = 1e-5):
      self.gamma = gamma
      self.epsilon = epsilon
      self.learning_rate = learning_rate
      self.eps_reduc = eps_reduc
      self.eps_min = eps_min
      self.action_space = [i for i in range(numof_actions)]
      self.mem_size = max_mem_size
      self.batch_size = batch_size
      self.mem_counter = 0

      self.Q_net = DQN(in_dims = in_dims, fc1_dims = 256, fc2_dims = 256, fc3_dims = 256,
                       out_dims = numof_actions, learning_rate=self.learning_rate)

      self.state_mem = np.zeros((self.mem_size, *in_dims), dtype = np.float32)
      self.new_state_mem = np.zeros((self.mem_size, *in_dims), dtype = np.float32)
      self.action_mem = np.zeros(self.mem_size, dtype = np.int32)
      self.reward_mem = np.zeros(self.mem_size, dtype = np.float32)
      self.terminal_mem = np.zeros(self.mem_size, dtype = np.bool)

 # defining function in order to store the transitions of the environment:


    def store_trans(self, state, action, reward, new_state, finished):
      idx = self.mem_counter % self.mem_size
      self.state_mem[idx] = state
      self.new_state_mem[idx] = new_state
      self.reward_mem[idx] = reward
      self.action_mem[idx] = action
      self.terminal_mem[idx] = finished

      self.mem_counter +=1

 # defining function in order to plan the next action:


    def plan_action(self, snap):
      # exploitation - exploration dilemma:
      if np.random.random() > self.epsilon:
        state = torch.tensor([snap]).to(self.Q_net.device)
        actions = self.Q_net.forward(state)
        action = torch.argmax(actions).item()
      else:
        action = np.random.choice(self.action_space)

      return action

  # defining the function in order to learn:


    def learn(self):

      if self.mem_counter < self.batch_size:
        return
      self.Q_net.optimizer.zero_grad()

      # getting the number of experiences:
      max_mem = min(self.mem_counter, self.mem_size)
      # choosing randomly some of the experiences (replace = False in order to not get the same experiences)
      batch = np.random.choice(max_mem, self.batch_size, replace = False)
      # indexing those experiences:
      batch_idx = np.arange(self.batch_size, dtype=np.int32)
      # taking the state, new_state, action, reward and terminal experiences based on the batch_idx
      state_value = torch.tensor(self.state_mem[batch]).to(self.Q_net.device)
      new_state_value = torch.tensor(self.new_state_mem[batch]).to(self.Q_net.device)
      action_value = self.action_mem[batch]
      reward_value = torch.tensor(self.reward_mem[batch]).to(self.Q_net.device)
      terminal_value = torch.tensor(self.terminal_mem[batch]).to(self.Q_net.device)
      # evaluating the q-value via the forward call:
      q = self.Q_net.forward(state_value)[batch_idx, action_value]
      q_next = self.Q_net.forward(new_state_value)
      # defining the value 0 for every terminal state:
      q_next[terminal_value] = 0

      # Bellman equation for q-values:
      q_value = reward_value + self.gamma * torch.max(q_next, dim=1)[0]
      # calculating the MSE:
      loss = self.Q_net.loss(q_value, q).to(self.Q_net.device)
      loss.backward()
      self.Q_net.optimizer.step()
      # linear reduction of epsilon:
      self.epsilon = self.epsilon - self.eps_reduc if self.epsilon > self.eps_min \
                        else self.eps_min

# ======================================================================================================================

# defining the agent and the hyperparameters:


agent = Agent(gamma=0.999, epsilon=1, learning_rate=0.001, in_dims=[8], batch_size=50, numof_actions=4)
scores = []
epsilon_values = []
n_episodes = 5000

# training the agent:

start_time = time.time()
for i in range(n_episodes):
  score = 0
  finished = False
  snap = env.reset()
  while not finished:
    action = agent.plan_action(snap)
    new_snap, reward, finished, info = env.step(action)
    score = score + reward
    agent.store_trans(snap,action,reward, new_snap, finished)
    agent.learn()
    snap = new_snap
  scores.append(score)
  epsilon_values.append(agent.epsilon)
  #getting the mean of the last 100 scores:
  mean_score = np.mean(scores[-100:])
  # printing simple statistics:
  print('Episode:',i+1, '|| Score: %.2f' % score, '|| Average Score: % .2f' % mean_score)
end_time = time.time()
mins, secs = done_time(start_time, end_time)
print(f'done in:, {mins}m {secs}s')