import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------- Define Neural Netwok Architecture --------------------------------
# Network inheritence from nn.Module, we can directly add stuff to the nn.Module and call Network
class Network(nn.Module):
    def __init__(self, lVl, lAl):
        super(Network, self).__init__()
        self.lVl, self.lAl = lVl, lAl
        # Let network be three fully connected layers
        self.fc1 = nn.Linear(lVl, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, lAl)

    def setas(self, anotherNetwork):
        self.lVl, self.lAl = anotherNetwork.lVl, anotherNetwork.lAl
        self.fc1, self.fc2, self.fc3 = anotherNetwork.fc1, anotherNetwork.fc2, anotherNetwork.fc3

    # state here defined by a particular instance of the Variables
    # ReLU(x) = max(0, x) # For each states instance, it will use nn to return Qval for each action.
    def approxQ(self, state):
        valLayer1 = F.relu(self.fc1(state))
        valLayer2 = F.relu(self.fc2(valLayer1))
        Qval = self.fc3(valLayer2)
        return Qval


class Window:
    def __init__(self, size):
        self.capacity = size
        self.count = 0
        self.values = torch.zeros(size)
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
        self.S = torch.zeros((capacity, lVl))
        self.S_ = torch.zeros((capacity, lVl))
        self.A = torch.zeros((capacity, 1), dtype=torch.int64)
        self.R = torch.zeros((capacity, 1))
        self.count = 0

    # each event is a tensor object
    def push(self, s, s_, a, r):
        replaceIndex = self.count % self.capacity
        self.S[replaceIndex, :] = s
        self.S_[replaceIndex, :] = s_
        self.A[replaceIndex, 0] = a
        self.R[replaceIndex, 0] = r
        self.count += 1

    # sample k batch from the memory
    def sample(self, batchSize):
        indexes = torch.randint(0, min(self.count, self.capacity), size=(batchSize,))
        # [s1,s2,...,sn],[s1_,s2_,...,sn_],[a1,a2,...,an],[r1,r2,...,rn]
        return self.S[indexes, :], self.S_[indexes, :], self.A[indexes, :], self.R[indexes, :]


class Dqn:
    def __init__(self, sVars, aSpace, mem, gamma=0.99, rWindowSize=10000, sampleSize=10000, beta=100):
        self.sVars = sVars
        self.aSpace = aSpace
        self.lVl, self.lAl = len(sVars), len(aSpace)
        self.gamma = gamma
        self.reward_window = Window(rWindowSize)  # use to evaluate the improvement
        self.model = Network(self.lVl, self.lAl)
        self.fixModel = Network(self.lVl, self.lAl)
        self.fixModel.setas(self.model)
        self.memory = mem
        self.nlearn = 0
        # Adam optimizer with 0.001 as learning rate. Bigger values would change Qvalue very drastically at every time step.
        # lr act as a smoothing factor for the Qvalue to update slowly.
        self.optimizer = optim.Adam(self.model.parameters(), lr=random.random() * 0.01)
        self.sampleSize = sampleSize
        self.beta = beta  # exploit factor (greediness)

    # beta is greediness, the higher the beta the less exploration
    def select_action(self, state):
        probs = F.softmax(self.model.approxQ(state) * self.beta, dim=0)
        # multinomial returns a torch object, item() turn it into a scalar object
        action = probs.multinomial(num_samples=1).item()
        return action

    def randOptimizerLr(self):
        return optim.Adam(self.model.parameters(), lr=random.random() * 0.01)

    def learn(self):
        if self.memory.count < self.sampleSize:
            return

        self.nlearn += 1
        if self.nlearn % 100 == 0:
            self.fixModel.setas(self.model)
        batchS, batchS_, batchA, batchR = self.memory.sample(self.sampleSize)
        # approxQ(batchS(batchSize, lVl)) -> (batchSize, lAl).gather(1,batchA) -> (batchSize,1).squeeze() = (batchSize)
        Qs = self.model.approxQ(batchS).gather(1, batchA).squeeze()  # gather keep value only for the action selected
        Vs_ = self.fixModel.approxQ(batchS_).detach().max(1).values  # detach: remove grad_fn, then max_a' Q(s',a')
        target = batchR.squeeze() + self.gamma * Vs_
        # smoothL1 is a combination of L1 and L2 loss, less sensitive to outliers than L2
        td_loss = F.smooth_l1_loss(Qs, target)
        # zero out -> calculate gradient based on td_loss -> update value (via gradient descent)
        self.optimizer.zero_grad()  # gradients are accumulated in pyTorch, manually zero out for a new pass is necessary
        td_loss.backward(retain_graph=True)  # performs backward pass, True: perform multi-pass for multi-variables
        self.optimizer.step()  # step of an optimizer: use to update the value based on the grad_fn
        # update learning rate

    def update(self, s, s_, a, r):  # r(s)
        # add memory and history
        self.memory.push(s, s_, a, r)
        self.reward_window.push(r)

    def score(self):
        return self.reward_window.mean()

    def save(self, name=""):
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            "AI/AI" + str(name) + ".pth",
        )

    def load(self, name=""):
        if os.path.isfile("AI/AI" + str(name) + ".pth"):
            print("=> loading checkpoint... ")
            checkpoint = torch.load("AI/AI" + str(name) + ".pth")
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            print("done !")
        else:
            print("no checkpoint found...")
