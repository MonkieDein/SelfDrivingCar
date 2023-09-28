import math
import os
import random
from itertools import chain

import numpy as np
import pandas as pd
import pygame
import torch

from dqn import Dqn


def vec(tup):
    return np.array(tup)


def vel(speed, angle):
    return speed * vec((math.cos(math.radians(angle)), -math.sin(math.radians(angle))))


def L2(vec):
    return math.sqrt(sum(vec**2))


def u1(vec):
    return vec / L2(vec)


def get4corners(loc, size, angle=0):
    return (
        tuple(loc),
        tuple(loc + vel(size[0], angle)),
        tuple(loc + vel(size[0], angle) + vel(size[1], angle - 90)),
        tuple(loc + vel(size[1], angle - 90)),
    )


def makeLinePair(corners):
    return [(p, corners[i - 1]) for i, p in enumerate(corners)]


class Map:
    def __init__(self, folder="", screenSize=[1280, 720]):
        self.inCorner, self.outCorner, self.cps = self.load(folder, screenSize)
        self.lCPl = len(self.cps)
        self.wallLines = makeLinePair(self.inCorner) + makeLinePair(self.outCorner)

    def draw(self, screen, inFill="orange", outFill="black", linesize=3, inLine="red", outLine="red"):
        pygame.draw.polygon(screen, outFill, self.outCorner)
        pygame.draw.polygon(screen, inFill, self.inCorner)
        pygame.draw.polygon(screen, outLine, self.outCorner, width=linesize)
        pygame.draw.polygon(screen, inLine, self.inCorner, width=linesize)

    def save(self, folder):
        if not os.path.exists(folder):
            os.mkdir(folder)
        inDf = pd.DataFrame(self.inCorner, columns=["x", "y"])
        outDf = pd.DataFrame(self.outCorner, columns=["x", "y"])
        cpDf = pd.DataFrame(self.cps, columns=["x1", "y1", "x2", "y2"])
        inDf.to_parquet(os.path.join(folder, "inDf.parquet"))
        outDf.to_parquet(os.path.join(folder, "outDf.parquet"))
        cpDf.to_parquet(os.path.join(folder, "cpDf.parquet"))

    def load(self, folder, screenSize):
        if folder:
            if os.path.exists(folder):
                return (
                    pd.read_parquet(os.path.join(folder, "inDf.parquet")).values.tolist(),
                    pd.read_parquet(os.path.join(folder, "outDf.parquet")).values.tolist(),
                    pd.read_parquet(os.path.join(folder, "cpDf.parquet")).values.tolist(),
                )
            else:
                print("Folder does not exist, Please set up a map")
        return self.setup_map(screenSize)

    def setup_map(self, screenSize):
        pygame.init()
        screen = pygame.display.set_mode(screenSize)
        pygame.display.set_caption("Car Race Map setup")
        screenW, screenH = screenSize
        setup = "OutCorner"  # Checkpoint, Inner corner, Outer corner
        innerCorner = []
        outerCorner = []
        checkpoint = []
        cur_set = outerCorner

        while setup:
            screen.fill("orange")
            instruction = pygame.font.Font(None, 15).render(
                "L-click to set, R-click to undo, Space to Confirm, setting " + setup, True, "purple"
            )
            screen.blit(instruction, (screenW // 100, screenH // 100))
            mx, my = pygame.mouse.get_pos()
            pygame.time.wait(100)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    setup = ""
                if event.type == pygame.KEYUP and event.key == pygame.K_SPACE:
                    if setup == "OutCorner":
                        cur_set, setup = innerCorner, "InCorner"
                    elif setup == "InCorner":
                        cur_set, setup = checkpoint, "CheckPoint"
                    elif setup == "CheckPoint":
                        cur_set, setup = [], ""
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        cur_set.append([mx, my])
                    if event.button == 3:
                        if len(cur_set):
                            del cur_set[-1]

            if len(outerCorner):
                if len(outerCorner) > 1:
                    preview = [[mx, my]] if setup == "OutCorner" else []
                    pygame.draw.polygon(screen, "black", outerCorner + preview)
                    pygame.draw.polygon(screen, "red", outerCorner + preview, width=1)
                elif len(outerCorner):
                    pygame.draw.circle(screen, "red", outerCorner[0], 2)

            if len(innerCorner):
                if len(innerCorner) > 1:
                    preview = [[mx, my]] if setup == "InCorner" else []
                    pygame.draw.polygon(screen, "orange", innerCorner + preview)
                    pygame.draw.polygon(screen, "red", innerCorner + preview, width=1)
                else:
                    pygame.draw.circle(screen, "red", innerCorner[0], 2)

            if setup == "CheckPoint":
                for i in range(0, len(checkpoint) - 1, 2):
                    pygame.draw.line(screen, "white", checkpoint[i], checkpoint[i + 1])
                if len(checkpoint) % 2:
                    pygame.draw.line(screen, "white", checkpoint[-1], [mx, my])

            pygame.draw.circle(screen, "white", [mx, my], 3)
            pygame.display.update()

        pygame.quit()
        return innerCorner, outerCorner, [checkpoint[i] + checkpoint[i + 1] for i in range(0, len(checkpoint), 2)]


class Car:
    def __init__(self, x, y, w, h, img, map, speed=0.0, angle=0, r0=0, cp=0, cumR=0):
        self.pos = vec((x, y))
        self.w = w
        self.h = h
        self.speed = speed
        self.angle = angle
        self.img = img
        self.map = map
        self.reward = r0
        self.cp = cp
        self.cumR = cumR
        self.mapRange = self.get_mapRange()
        self.center = self.getCarCenter(self.pos)
        self.sensDir = self.sensorDirection()
        self.sensDist = self.getSensors()

    def randomInit(self):
        pos_i = random.randint(0, self.map.lCPl - 1)
        self.pos = vec(
            (
                0.5 * (self.map.cps[pos_i][0] + self.map.cps[pos_i][2]),
                0.5 * (self.map.cps[pos_i][1] + self.map.cps[pos_i][3]),
            )
        )
        self.speed = random.randint(0, 10) / 10.0
        self.angle = random.randint(0, 119) * 3
        self.reward = 0
        self.cp = (pos_i + 1) % self.map.lCPl
        self.cumR = 0
        self.mapRange = self.get_mapRange()
        self.updateSensor()

    def reset(self, initial):
        self.pos = initial.pos.copy()
        self.w = initial.w
        self.h = initial.h
        self.speed = initial.speed
        self.angle = initial.angle
        self.img = initial.img
        self.map = initial.map
        self.reward = initial.reward
        self.cp = initial.cp
        self.cumR = initial.cumR
        self.mapRange = self.get_mapRange()
        self.updateSensor()

    def copy(self):
        return Car(
            self.pos[0],
            self.pos[1],
            self.w,
            self.h,
            self.img,
            self.map,
            speed=self.speed,
            angle=self.angle,
            r0=self.reward,
            cp=self.cp,
            cumR=self.cumR,
        )

    def get_mapRange(self):
        return np.array(
            [
                max([i[0] for i in self.map.outCorner]) - min([i[0] for i in self.map.outCorner]),
                max([i[1] for i in self.map.outCorner]) - min([i[1] for i in self.map.outCorner]),
            ]
        )

    def get_input(self):
        keys_pressed = pygame.key.get_pressed()
        if keys_pressed[pygame.K_LEFT]:
            self.angle = (self.angle + 3) % 360
        if keys_pressed[pygame.K_RIGHT]:
            self.angle = (self.angle - 3) % 360
        if keys_pressed[pygame.K_UP]:
            self.speed = self.speed + 0.1
        if keys_pressed[pygame.K_DOWN]:
            self.speed = max(0.0, self.speed - 0.1)

    def getAIstate(self):
        return (
            list(vec(self.sensDist) / max(self.mapRange))
            + [self.speed]
            + [self.angle / 360.0]
            + list(self.pos / self.mapRange)
            + [self.cp]
        )

    def get_AI_input(self, keys_pressed):
        if "L" in keys_pressed:
            self.angle = (self.angle + 3) % 360
        if "R" in keys_pressed:
            self.angle = (self.angle - 3) % 360
        if "U" in keys_pressed:
            self.speed = self.speed + 0.1
        if "D" in keys_pressed:
            self.speed = max(0.0, self.speed - 0.1)

    def getPath(self, prevCorner):
        return [(p1, p2) for p1, p2 in zip(prevCorner, self.pos2CarCorners(self.pos + vel(self.speed, self.angle)))]

    def update(self, prevCorner):
        collide, portion = getCollision(self.getPath(prevCorner), self.map.wallLines)
        self.reward = -0.01
        while getCollision(self.getPath(prevCorner), [(self.map.cps[self.cp][0:2], self.map.cps[self.cp][2:])])[0]:
            self.reward += 0.1
            self.cp = (self.cp + 1) % self.map.lCPl
        self.cumR += self.reward
        self.pos += portion * vel(self.speed, self.angle)
        self.updateSensor()
        return collide

    def getCarCenter(self, pos):
        return pos + vec(self.img.get_rect().center)

    def getCarCorners(self, center):
        carUL = center - (vel(self.w, self.angle) + vel(self.h, self.angle - 90)) / 2
        return get4corners(carUL, vec((self.w, self.h)), self.angle)

    def pos2CarCorners(self, pos):
        return self.getCarCorners(self.getCarCenter(pos))

    def cornerUnitVector(self):
        center = self.center
        return [u1(center - p) for p in self.getCarCorners(self.center)]

    def sensorDirection(self):
        return self.cornerUnitVector() + [vel(1, self.angle + a) for a in np.arange(0, 360, 90)]

    def sensorDistance(self, center, ray, wallLines):
        return min(
            t
            for t, u in [xFactor(center, center + ray, p3, p4) for p3, p4 in wallLines]
            if t >= 0 and not math.isinf(t) and 0 <= u <= 1
        )

    def getSensors(self):
        distances = [self.sensorDistance(self.center, dir, self.map.wallLines) for dir in self.sensDir]
        return distances

    def updateSensor(self):
        self.center = self.getCarCenter(self.pos)
        self.sensDir = self.sensorDirection()
        self.sensDist = self.getSensors()

    def drawSensor(self, screen):
        for ray, dist in zip(self.sensDir, self.sensDist):
            pygame.draw.line(screen, (30, 30, 30), self.center, self.center + dist * ray, 1)
            pygame.draw.circle(screen, "white", self.center + dist * ray, 1)

    def draw(self, screen):
        rotated_image = pygame.transform.rotate(self.img, self.angle)
        rotated_rect = rotated_image.get_rect(center=self.img.get_rect().center)
        screen.blit(rotated_image, tuple(self.pos + vec(rotated_rect.topleft)))
        # draw the edges of the car object
        pygame.draw.polygon(screen, "green", self.getCarCorners(self.center), 1)
        self.drawSensor(screen)


def draw_window(screen, cars, map):
    screen.fill("orange")
    map.draw(screen)
    for car in cars:
        car.draw(screen)
    pygame.display.update()


# p_intersect = p1 + u * (p2-p1)
def xFactor(p1, p2, p3, p4):
    x1, x2, x3, x4 = p1[0], p2[0], p3[0], p4[0]
    y1, y2, y3, y4 = p1[1], p2[1], p3[1], p4[1]
    Num_t = ((x1 - x3) * (y3 - y4)) - ((y1 - y3) * (x3 - x4))
    Num_u = ((x1 - x3) * (y1 - y2)) - ((y1 - y3) * (x1 - x2))
    denominator = ((x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4))
    if denominator == 0:
        return (float("inf"), float("inf"))
    t = Num_t / denominator
    u = Num_u / denominator
    return (t, u)


# Two Line intersections L1 (p1,p2) and L2 (p3,p4)
def isIntersect(p1, p2, p3, p4):
    t, u = xFactor(p1, p2, p3, p4)
    return (0 <= t <= 1) and (0 <= u <= 1)


def getCollision(carPaths, wallLines):
    collide, time = False, 1.0
    for p1, p2 in carPaths:
        intersect = [xFactor(p1, p2, p3, p4) for p3, p4 in wallLines]
        for t, u in intersect:
            if (0 <= t <= 1) and (0 <= u <= 1):
                collide, time = True, min(t, time)
    return collide, time


def game_over_display(screen, screenW, screenH):
    display_game_over = pygame.font.Font(None, 90).render("Game Over", True, "purple")
    screen.blit(display_game_over, (screenW // 3, screenH // 2))
    display_restart = pygame.font.Font(None, 45).render("Space to Restart", True, "purple")
    screen.blit(display_restart, (screenW // 3, screenH // 3))
    pygame.display.update()
