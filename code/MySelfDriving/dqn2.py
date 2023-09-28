import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import tensor


# ------------------- Define Neural Netwok Architecture --------------------------------
# Network inheritence from nn.Module, we can directly add stuff to the nn.Module and call Network
class Network(nn.Module):
    def __init__(self, lVl, lAl):
        super(Network, self).__init__()
        self.lVl, self.lAl = lVl, lAl
        # Let network be three fully connected layers
        self.fc1 = nn.Linear(lVl, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, lAl)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def np2dev(self, obj):
        return tensor(obj).to(self.device)

    # state here defined by a particular instance of the Variables
    # ReLU(x) = max(0, x) # For each states instance, it will use nn to return Qval for each action.
    def approxQ(self, state):
        valLayer1 = F.relu(self.fc1(state))
        valLayer2 = F.relu(self.fc2(valLayer1))
        qVal = self.fc3(valLayer2)
        return qVal


class Window:
    def __init__(self, size):
        self.capacity = size
        self.count = 0
        self.values = np.zeros(size, dtype=np.float32)
        pass

    def push(self, val):
        replaceIndex = self.count % self.capacity
        self.values[replaceIndex] = val
        self.count += 1

    def mean(self):
        if self.count == 0:
            return 0
        elif self.count < self.capacity:
            return self.values[0 : self.count].mean()
        else:
            return self.values.mean()


# ------------------- Experience Replay --------------------------------
# Memory consist of (s,s_,a,r)
class ReplayMemory:
    def __init__(self, lVl, capacity=100000):
        self.capacity = capacity
        self.S = np.zeros((capacity, lVl), dtype=np.float32)
        self.S_ = np.zeros((capacity, lVl), dtype=np.float32)
        self.A = np.zeros((capacity, 1), dtype=np.int64)
        self.R = np.zeros((capacity, 1), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=bool)
        self.count = 0

    # each event is a tensor object
    def push(self, s, s_, a, r, done, n=1):
        replaceIndex = self.count % self.capacity
        n2 = 0
        if (replaceIndex + n) > self.capacity:
            n2 = (replaceIndex + n) % self.capacity
            n = self.capacity - replaceIndex
        self.S[replaceIndex : (replaceIndex + n), :] = s[:n]
        self.S_[replaceIndex : (replaceIndex + n), :] = s_[:n]
        self.A[replaceIndex : (replaceIndex + n), 0] = a[:n]
        self.R[replaceIndex : (replaceIndex + n), 0] = r[:n]
        self.done[replaceIndex : (replaceIndex + n), 0] = done[:n]
        self.count += n
        if n2 != 0:
            self.push(s[n:], s_[n:], a[n:], r[n:], done[n:], n=n2)

    # sample k batch from the memory
    def sample(self, batchSize):
        indexes = np.random.choice(min(self.count, self.capacity), batchSize, replace=False)
        # [s1,s2,...,sn],[s1_,s2_,...,sn_],[a1,a2,...,an],[r1,r2,...,rn]
        return self.S[indexes], self.S_[indexes], self.A[indexes], self.R[indexes], self.done[indexes]


class Dqn:
    def __init__(self, sVars, aSpace, mem, gamma=0.99, rWindow=10000, sampleSize=10000, cntTarget=100, n=0, eps=0.2):
        self.sVars = sVars
        self.aSpace = aSpace
        self.lVl, self.lAl = len(sVars), len(aSpace)
        self.gamma = gamma
        self.reward_window = Window(rWindow)  # use to evaluate the improvement
        self.model = Network(self.lVl, self.lAl)
        self.targetNet = Network(self.lVl, self.lAl)
        self.cntTarget = cntTarget
        self.memory = mem
        self.nlearn = n
        # Adam optimizer with 0.001 as learning rate. Bigger values would change Qvalue very drastically at every time step.
        # lr act as a smoothing factor for the Qvalue to update slowly.
        self.sampleSize = sampleSize
        self.epsNum = 0.0 if n == 1 else n
        self.eps = eps  # epsilon greedy

    # beta is greediness, the higher the beta the less exploration
    def select_action(self, observation):
        # action = F.softmax(self.model.approxQ(state), dim=0).multinomial(num_samples=1).item()
        with torch.no_grad():
            if np.random.random() > max(self.eps, self.epsNum / self.nlearn):
                qvals = self.model.approxQ(observation.to(self.model.device))
                action = torch.argmax(qvals).item()
            else:
                action = np.random.choice(np.arange(self.lAl))
            return action

    def update_target_network(self):
        self.nlearn += 1
        if self.nlearn % self.cntTarget == 0:
            self.targetNet.load_state_dict(self.model.state_dict())

    def soft_update_target_network(self, update_rate=0.05):
        # Soft update model parameters
        self.nlearn += 1
        if self.nlearn % self.cntTarget == 0:
            for target_param, param in zip(self.targetNet.parameters(), self.model.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - update_rate) + param.data * update_rate)

    def learn(self):
        if self.memory.count < self.sampleSize:
            return
        self.model.optimizer.zero_grad()  # gradients are accumulated in pyTorch, manually zero out ->  for a new pass is necessary
        self.soft_update_target_network()
        batchS, batchS_, batchA, batchR, batchDone = self.sample_memory()
        # approxQ(batchS(batchSize, lVl)) -> (batchSize, lAl).gather(1,batchA) -> (batchSize,1).squeeze() = (batchSize)
        Qs = self.model.approxQ(batchS).gather(1, batchA).squeeze()  # gather keep value only for the action selected
        Vs_ = self.targetNet.approxQ(batchS_).detach().max(1).values  # detach: remove grad_fn, then max_a' Q(s',a')
        Vs_[batchDone.squeeze()] = 0.0
        targetVal = batchR.squeeze() + self.gamma * Vs_
        # calculate gradient based on td_loss -> update value (via gradient descent)
        td_loss = self.model.loss(Qs, targetVal)
        td_loss.backward()  # performs backward pass, True: perform multi-pass for multi-variables
        self.model.optimizer.step()  # step of an optimizer: use to update the value based on the grad_fn

    def store_memory(self, s, s_, a, r, done):  # r(s)
        # add memory and history
        self.memory.push(s, s_, a, r, done)
        self.reward_window.push(r)

    def sample_memory(self):
        batchS, batchS_, batchA, batchR, batchDone = self.memory.sample(self.sampleSize)
        S = self.model.np2dev(batchS)
        S_ = self.model.np2dev(batchS_)
        A = self.model.np2dev(batchA)
        R = self.model.np2dev(batchR)
        Done = self.model.np2dev(batchDone)
        return S, S_, A, R, Done

    def score(self):
        return self.reward_window.mean()

    def save(self, name=""):
        torch.save(self.model.state_dict(), "AI/AI" + str(name) + ".pth")
        torch.save(self.targetNet.state_dict(), "AI/AI" + str(name) + "target.pth")

    def load(self, name=""):
        if os.path.isfile("AI/AI" + str(name) + ".pth"):
            print("=> loading checkpoint... ")
            self.model.load_state_dict(torch.load("AI/AI" + str(name) + ".pth"))
            if os.path.isfile("AI/AI" + str(name) + "target.pth"):
                self.targetNet.load_state_dict(torch.load("AI/AI" + str(name) + "target.pth"))
            print("done !")
        else:
            print("no checkpoint found...")
