import math
import os
import random
from itertools import chain

import numpy as np
import pandas as pd
import pygame
import torch
from torch import tensor

#####################################################################################
################################ GEOMETRY SECTIONS ##################################
#####################################################################################

ACT = ["", "L", "R", "U", "D", "UL", "UR", "DL", "DR"]


# vec has dimension [...,2] where the norm function is used on coordinate x and y.
def L2(vec):
    return torch.norm(vec, dim=-1).unsqueeze(-1)


# vec has dimension [...,2] where the u1 function is used on coordinate x and y.
def u1(vec):
    return vec / L2(vec)


# speed and angle dimension should be [|CAR|,...] : return [|CAR|,...,|coord|]
def vel(speed, angle):
    return speed * torch.stack((torch.cos(torch.deg2rad(angle)), -torch.sin(torch.deg2rad(angle))), dim=-1)


# upperLefts has dimension [...,2], return corners coordinate [..., 4,2]
def get4corners(upperLeft, size, angle=0):
    X = vel(size[..., 0], angle)
    Y = vel(size[..., 1], angle - 90)
    corners = upperLeft.unsqueeze(-2) + torch.stack((torch.zeros_like(X), X, X + Y, Y), dim=-2)
    return corners


# corners should have dimension [...,N,|coord|], return linePairs with dimension [...,N-1,2,|coord|]
def makeLinePair(corners):
    return torch.stack((corners, torch.cat((corners[1:], corners[:1]))), dim=-2)


#####################################################################################
################################# PHYSICS SECTIONS ##################################
#####################################################################################


# Intersection of two lines. L1 has shape (M,1,2,2) and L2 has shape (1,N,2,2)
# the third [2] dimension refers to points, and forth [3] dimension refers to x,y
# return t, u that has [M,N] dimensions, note that M and N can be multi dimensional
def getIntersect(L1, L2):
    x1, x2, x3, x4 = L1[..., 0, 0], L1[..., 1, 0], L2[..., 0, 0], L2[..., 1, 0]
    y1, y2, y3, y4 = L1[..., 0, 1], L1[..., 1, 1], L2[..., 0, 1], L2[..., 1, 1]
    Num_t = ((x1 - x3) * (y3 - y4)) - ((y1 - y3) * (x3 - x4))
    Num_u = ((x1 - x3) * (y1 - y2)) - ((y1 - y3) * (x1 - x2))
    denominator = ((x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4))
    inf_tensor = torch.full_like(Num_t, float("inf"))
    t = torch.where(denominator == 0, inf_tensor, Num_t / denominator)
    u = torch.where(denominator == 0, inf_tensor, Num_u / denominator)
    return (t, u)


def intersect(t, u):
    return (0 <= t) & (t <= 1) & (0 <= u) & (u <= 1)


def getCollision(L1, L2):
    t, u = getIntersect(L1, L2)
    hasCollision = intersect(t, u)
    t[~hasCollision] = 1.0
    while hasCollision.dim() > 1:
        hasCollision = hasCollision.any(dim=-1)
        t = t.min(dim=-1).values
    return hasCollision, t


#####################################################################################
############################### MAP SETUP SECTIONS ##################################
#####################################################################################


class Map:
    def __init__(self, folder="", screenSize=[1280, 720]):
        self.inCorner, self.outCorner, self.cps = self.load(folder, screenSize)
        self.lCPl = len(self.cps)
        self.cpsT = tensor(self.cps, dtype=torch.float).reshape((self.lCPl, 2, 2))
        self.wallLines = torch.cat((makeLinePair(tensor(self.inCorner)), makeLinePair(tensor(self.outCorner))), dim=-3)
        self.midCps = self.cpsT.mean(-2)
        ox, oy = [i[0] for i in self.outCorner], [i[1] for i in self.outCorner]
        self.range = tensor([max(ox) - min(ox), max(oy) - min(oy)])

    def draw(self, screen, checkpoint=False, inFill="orange", outFill="black", linesize=3, inLine="red", outLine="red"):
        pygame.draw.polygon(screen, outFill, self.outCorner)
        pygame.draw.polygon(screen, inFill, self.inCorner)
        pygame.draw.polygon(screen, outLine, self.outCorner, width=linesize)
        pygame.draw.polygon(screen, inLine, self.inCorner, width=linesize)
        if checkpoint:
            [pygame.draw.line(screen, "grey", p1.tolist(), p2.tolist()) for p1, p2 in self.cpsT]

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
                        else:
                            if setup == "CheckPoint":
                                cur_set, setup = innerCorner, "InCorner"
                            elif setup == "InCorner":
                                cur_set, setup = outerCorner, "OutCorner"

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


class Cars:
    def __init__(self, lCl, w, h, img, map: Map, r0=None, cumR=None, speed=None, angle=None, cp=None):
        self.img, self.map = img, map  # save image and map
        self.lCl = lCl  # save number of cars
        self.w, self.h = w, h  # car size
        self.reward = self.ifNoneSet2(r0, 0.0)  # reward received
        self.cumR = self.ifNoneSet2(cumR, 0.0)  # cumulative rewards
        self.speed = self.ifNoneSet2(speed, 0.0)  # speed
        self.angle = self.ifNoneSet2(angle, 0)  # direction of the car
        self.cp = self.ifNoneSet2(cp, 0)  # checkpoint index
        self.pos = self.map.midCps[(self.cp - 1) % self.map.lCPl, :]
        [self.center, self.sensDir, self.sensDist] = [None, None, None]
        self.updateSensor()

    def ifNoneSet2(self, var, value):
        if var is None:
            return torch.full((self.lCl,), value)
        else:
            return var

    def randomInit(self):
        self.cp = torch.randint(low=0, high=self.map.lCPl, size=(self.lCl,))
        self.pos = self.map.midCps[(self.cp - 1) % self.map.lCPl, :]
        self.speed = torch.randint(low=3, high=40, size=(self.lCl,)) / 10.0
        self.angle = torch.randint(low=0, high=120, size=(self.lCl,)) * 3
        self.reward *= 0
        self.cumR *= 0
        self.updateSensor()

    def reset(self):
        self.speed *= 0
        self.angle *= 0
        self.reward *= 0
        self.cumR *= 0
        self.cp *= 0
        self.updateSensor()

    def copy(self):
        # lCl, w, h, img, map:Map,r0=None,cumR=None,speed=None,angle=None,cp=None
        return Cars(
            self.lCl,
            self.w,
            self.h,
            self.img,
            self.map,
            r0=self.reward,
            cumR=self.cumR,
            speed=self.speed,
            angle=self.angle,
            cp=self.cp,
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
            self.speed = (self.speed - 0.1).clamp_min(0.3)

    def getAIstate(self):
        return torch.cat(
            (
                self.sensDist / max(self.map.range),
                self.speed.unsqueeze(1),
                self.angle.unsqueeze(1) / 360.0,
                self.pos / self.map.range,
                self.cp.unsqueeze(1) / self.map.lCPl,
            ),
            dim=1,
        )

    # action has dims (|Cars|,|action|)
    def get_AI_input(self, action, game_overs):
        id = torch.arange(self.lCl)[~game_overs]
        action = action[~game_overs]
        self.angle[id[action[:, 0]]] = (self.angle[id[action[:, 0]]] + 3) % 360
        self.angle[id[action[:, 1]]] = (self.angle[id[action[:, 1]]] - 3) % 360
        self.speed[id[action[:, 2]]] = self.speed[id[action[:, 2]]] + 0.1
        self.speed[id[action[:, 3]]] = (self.speed[id[action[:, 3]]] - 0.1).clamp_min(0.1)

    def getCarCenter(self, pos):
        return pos + tensor(self.img.get_rect().center)

    def getCarCorners(self, center):
        carUL = center - (vel(self.w, self.angle) + vel(self.h, self.angle - 90)) / 2
        return get4corners(carUL, tensor([self.w, self.h]), self.angle)

    def pos2CarCorners(self, pos):
        return self.getCarCorners(self.getCarCenter(pos))

    # (Cars,Corner,2,2)
    def projectPath(self, prevCorner):
        return torch.stack(
            (prevCorner, self.pos2CarCorners(self.pos + vel(self.speed.unsqueeze(-1), self.angle))), dim=-2
        )

    # (Cars, Corner, 2)
    def cornerUnitVector(self):
        return u1(self.center.unsqueeze(-2) - self.getCarCorners(self.center))

    # (Cars, sensors, 2)
    def sensorDirection(self):
        return torch.cat((self.cornerUnitVector(), vel(1, self.angle.unsqueeze(-1) + torch.arange(0, 360, 90))), dim=-2)

    # self center is (Cars,coord), sensDir is (Cars,Sens,coord), second last dim is a pair (center, center+sensDir)
    def getSensors(self):
        # L1 is (Cars, Sens, WallLines, Pairs, Coord)
        L1 = (
            self.center.unsqueeze(-2).unsqueeze(-2)
            + torch.stack((torch.zeros_like(self.sensDir), self.sensDir), dim=-2)
        ).unsqueeze(-3)
        t, u = getIntersect(L1, self.map.wallLines)
        t[~((t >= 0) & (0 <= u) & (u <= 1))] = float("inf")
        # return distance has dimension of (Cars,Sens)
        return t.min(dim=-1).values

    def updateSensor(self):
        self.center = self.getCarCenter(self.pos)
        self.sensDir = self.sensorDirection()
        self.sensDist = self.getSensors()

    # previous corner refers to prevCorner that is not done
    def update(self, prevCorner, done):
        cornerPath = self.projectPath(prevCorner)
        id = torch.arange(self.lCl)[~done]
        collide, portion = getCollision(cornerPath[id].unsqueeze(-3), self.map.wallLines)
        self.reward[id] = self.reward[id] * 0 - 0.001  # reward if no checkpoint is passed = -0.001
        passChp = getCollision(cornerPath[id], self.map.cpsT[self.cp[id]].unsqueeze(-3))[0]
        while any(passChp):
            # print("Checkpoint passing")
            id = id[passChp]
            self.reward[id] += 0.1
            self.cp[id] = (self.cp[id] + 1) % self.map.lCPl
            passChp = getCollision(cornerPath[id], self.map.cpsT[self.cp[id]].unsqueeze(-3))[0]

        self.cumR[~done] += self.reward[~done]
        self.pos[~done] += portion.unsqueeze(-1) * vel(self.speed[~done].unsqueeze(-1), self.angle[~done])
        self.updateSensor()
        return collide

    def drawSensor(self, screen, line=True, dot=True):
        for carMid, rays, dists in zip(self.center, self.sensDir, self.sensDist):
            for ray, dist in zip(rays, dists):
                if line:
                    pygame.draw.line(screen, (30, 30, 30), carMid.tolist(), (carMid + (dist * ray)).tolist(), 1)
                if dot:
                    pygame.draw.circle(screen, "white", (carMid + (dist * ray)).tolist(), 1)

    def draw(self, screen, sensor=True, edge=True):
        rotated_image = [pygame.transform.rotate(self.img, a) for a in self.angle]
        center = self.img.get_rect().center
        topLeftCoord = self.pos + tensor([i.get_rect(center=center).topleft for i in rotated_image])
        [screen.blit(img, pos.tolist()) for img, pos in zip(rotated_image, topLeftCoord)]
        # draw the edges of the car object
        if edge:
            [pygame.draw.polygon(screen, "green", car.tolist(), 1) for car in self.getCarCorners(self.center)]
        if sensor:
            self.drawSensor(screen)


def draw_window(screen, cars, map, it, W, H):
    screen.fill("orange")
    map.draw(screen, checkpoint=False)
    screen.blit(pygame.font.Font(None, 90).render(f"iteration {it}", True, "green"), (W / 2 - 200, H / 2 - 45))
    cars.draw(screen, sensor=False, edge=False)
    pygame.display.update()


def game_over_display(screen, screenW, screenH):
    display_game_over = pygame.font.Font(None, 90).render("Game Over", True, "purple")
    screen.blit(display_game_over, (screenW // 3, screenH // 2))
    display_restart = pygame.font.Font(None, 45).render("Space to Restart", True, "purple")
    screen.blit(display_restart, (screenW // 3, screenH // 3))
    pygame.display.update()
