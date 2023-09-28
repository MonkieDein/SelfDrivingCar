import math
import os
import random
from itertools import chain

import numpy as np
import pandas as pd
import pygame
import torch

from domain2 import *
from dqn2 import Dqn, Network, ReplayMemory

VAR = ["sens1", "sens2", "sens3", "sens4", "sens5", "sens6", "sens7", "sens8", "vx", "vy", "x", "y", "cp"]
ACT = ["", "U", "D", "L", "R", "UL", "UR", "DL", "DR"]
tensorACT = tensor([["L" in a, "R" in a, "U" in a, "D" in a] for a in ACT])
lAl = len(ACT)
NCAR = 100
# currentBestNetworks = [Network(len(VAR), len(ACT)) for i in range(NCAR)]
# currentBestCumR = [-float("inf") for i in range(NCAR)]
mem = ReplayMemory(len(VAR), capacity=100000)
AIs = [Dqn(VAR, ACT, mem, n=1, eps=1e-6) for i in range(NCAR)]

for i, ai in enumerate(AIs):
    ai.load(i)

WIDTH, HEIGHT = 1280, 720
FPS = 60
ROAD = 90
CAR_H, CAR_W = 8, 24
showGame = False

map = Map("stage/simple", screenSize=[WIDTH, HEIGHT])

if showGame:
    pygame.init()
    screen = pygame.display.set_mode([WIDTH, HEIGHT])
    pygame.display.set_caption("Car Race Simulation")
    clock = pygame.time.Clock()

CAR_IMG = pygame.image.load(os.path.join("Assets", "car.png"))
CAR_IMG = pygame.transform.rotate(pygame.transform.scale(CAR_IMG, (CAR_H, CAR_W)), 270)
cars = Cars(NCAR, CAR_W, CAR_H, CAR_IMG, map)
cars.randomInit()
it = 0
run = True
game_overs = tensor([False for i in range(NCAR)])
done_playin = [False for i in range(NCAR)]
cumulativeReward = []
while run:
    if showGame:
        _ = clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

    if not all(game_overs):
        prevCorner = cars.getCarCorners(cars.center)

        S = cars.getAIstate()
        A = tensor([ai.select_action(s) for ai, s in zip(AIs, S)])  # torch.randint(low=0, high=lAl, size=(cars.lCl,)) #
        cars.get_AI_input(tensorACT[A], game_overs)
        # Calculate next reward r'(s')
        done = cars.update(prevCorner, game_overs)
        # if torch.isinf(cars.sensDist.sum()):
        #     run = False

        mem.push(
            S[~game_overs], cars.getAIstate()[~game_overs], A[~game_overs], cars.reward[~game_overs], done, n=len(done)
        )

        game_overs[~game_overs] = done

        for i in torch.randint(low=0, high=NCAR, size=(10,)):
            AIs[i].learn()

    if showGame:
        draw_window(screen, cars, map, it, WIDTH, HEIGHT)

    if all(game_overs):
        print(
            f"iters {it}, mean cumulative reward {cars.cumR.mean().item()}, max cumulative reward {cars.cumR.max().item()}"
        )
        cumulativeReward.append(cars.cumR.clone().detach())
        # for i, car in enumerate(cars):
        game_overs[:] = False
        cars.randomInit()
        for i, ai in enumerate(AIs):
            ai.save(i)

        it += 1
        if it > 1000:
            run = False

if showGame:
    pygame.quit()
