import math
import os
import random
from itertools import chain

import numpy as np
import pandas as pd
import pygame
import torch

from domain import *
from dqn2 import Dqn, Network, ReplayMemory

VAR = ["sens1", "sens2", "sens3", "sens4", "sens5", "sens6", "sens7", "sens8", "vx", "vy", "x", "y", "cp"]
ACT = ["", "U", "D", "L", "R", "UL", "UR", "DL", "DR"]
lAl = len(ACT)
NCAR = 100

currentBestNetworks = [Network(len(VAR), len(ACT)) for i in range(NCAR)]
currentBestCumR = [-float("inf") for i in range(NCAR)]

mem = ReplayMemory(len(VAR), capacity=100000)
AIs = [Dqn(VAR, ACT, mem) for i in range(NCAR)]

# for i, ai in enumerate(AIs):
#     ai.load(i)

WIDTH, HEIGHT = 1280, 720
FPS = 60
ROAD = 90
CAR_H, CAR_W = 8, 24
showGame = False

map = Map("stage/medium", screenSize=[WIDTH, HEIGHT])

if showGame:
    pygame.init()
    screen = pygame.display.set_mode([WIDTH, HEIGHT])
    pygame.display.set_caption("Car Race Simulation")
    clock = pygame.time.Clock()

CAR_IMG = pygame.image.load(os.path.join("Assets", "car.png"))
CAR_IMG = pygame.transform.rotate(pygame.transform.scale(CAR_IMG, (CAR_H, CAR_W)), 270)
initCar = Car(WIDTH / 8 + (ROAD - CAR_W) / 2, HEIGHT / 8 + (ROAD - CAR_H) / 2, CAR_W, CAR_H, CAR_IMG, map)
cars = [initCar.copy() for i in range(NCAR)]
for car in cars:
    car.randomInit()

it = 0
run = True
game_overs = [False for i in range(NCAR)]
done_playin = [False for i in range(NCAR)]

while run:
    if showGame:
        _ = clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            # if event.type == pygame.KEYUP:
            # if event.key == pygame.K_RETURN:
            # for i, ai in enumerate(AIs):
            #     ai.save(i)
    aA = np.random.choice(lAl, NCAR, replace=True)
    for i, car in enumerate(cars):
        game_over = game_overs[i]
        ai = AIs[i]
        if not game_over:
            s = car.getAIstate()
            prevCorner = car.getCarCorners(car.center)

            a = ai.select_action(torch.tensor(s, dtype=torch.float))  # a = random.choice(np.arange(lAl))  #
            car.get_AI_input(ACT[a])  # ai.aSpace[a])
            # Calculate next reward r'(s')

            game_overs[i] = car.update(prevCorner)

            ai.store_memory(s, car.getAIstate(), a, car.reward, game_overs[i])
            ai.learn()  # learn from batch memory

            if car.cumR > 300:
                game_overs[i] = True
                done_playin[i] = True

            if game_overs[i]:
                it += 1
                print(f"iters {it}, car{i}, cumulative reward {car.cumR}")
                if not done_playin[i]:
                    game_overs[i] = False
                    cars[i].randomInit()
                if currentBestCumR[i] < car.cumR:
                    currentBestCumR[i] = car.cumR
                    AIs[i].save(i)

    if showGame:
        # print("drawing")
        draw_window(screen, cars, map)
        # pygame.draw.line(screen, "red", map.cps[0][0:2], map.cps[0][2:])
        # pygame.display.update()
    if all(game_overs):
        if sum(done_playin) > 5:
            run = False

            # cars[i].reset(initCar)

            # if it % 10 == 0:
            #     for i, car in enumerate(cars):
            #         AIs[i].model.setas(currentBestNetworks[i])
            # print("reset")


if showGame:
    pygame.quit()
