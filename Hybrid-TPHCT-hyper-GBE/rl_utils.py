#!/usr/bin/python3

# Author(s): Luiz Felipe Vecchietti, Kyujin Choi, Taeyoung Kim
# Maintainer: Kyujin Choi (nav3549@kaist.ac.kr)
try:
    import _pickle as pickle
except:
    import pickle
import math
import numpy as np
import torch
import sys
import os
from torch.autograd import Variable

import matplotlib
import matplotlib.pyplot as plt

import helper

#reset_reason
NONE = 0
GAME_START = 1
SCORE_MYTEAM = 2
SCORE_OPPONENT = 3
GAME_END = 4
DEADLOCK = 5
GOALKICK = 6
CORNERKICK = 7
PENALTYKICK = 8
HALFTIME = 9
EPISODE_END = 10

#game_state
STATE_DEFAULT = 0
STATE_KICKOFF = 1
STATE_GOALKICK = 2
STATE_CORNERKICK = 3
STATE_PENALTYKICK = 4

#coordinates
MY_TEAM = 0
OP_TEAM = 1
BALL = 2
X = 0
Y = 1
Z = 2
TH = 3
ACTIVE = 4
TOUCH = 5
BALL_POSSESSION = 6
robot_size = 0.15

#robot_index
GK_INDEX = 0
D1_INDEX = 1
D2_INDEX = 2
F1_INDEX = 3
F2_INDEX = 4


USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LONG = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor



class TouchCounter(object):
    MaxCountMove = 10
    MaxCountGoal = 40

    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.reset()
    
    def reset(self):
        self.touch_count_move = np.zeros(self.num_agents)
        self.touch_count_goal = np.zeros(self.num_agents)
        self.once_touch = [False for _ in range(self.num_agents)]

    def isTouchedMoveCurrent(self):
        return [(self.touch_count_move[i] > 0) for i in range(self.num_agents)]

    def isTouchedGoalCurrent(self):
        return [(self.touch_count_goal[i] > 0) for i in range(self.num_agents)]

    def onceTouched(self):
        return self.once_touch
    
    def ShowTouchedInfo(self):
        helper.printConsole("touch count move: " + str(self.touch_count_move))
        helper.printConsole("touch count goal: " + str(self.touch_count_goal))

    def Counts(self, cur_posture, reset_reason):
        if not reset_reason == None:
            if reset_reason > 0:
                self.reset()

        touch_flag = [cur_posture[i][TOUCH] for i in range(self.num_agents)]

        for i in range(self.num_agents):
            if touch_flag[i]:
                self.touch_count_move[i] = self.MaxCountMove
                self.touch_count_goal[i] = self.MaxCountGoal
                self.once_touch[i] = True
            else:
                if self.touch_count_move[i] > 0:
                    self.touch_count_move[i] -= 1
                if self.touch_count_goal[i] > 0:
                    self.touch_count_goal[i] -= 1

        #return self.isTouchedMoveCurrent(), self.isTouchedGoalCurrent()


def predict_ball_velocity(cur_ball, prev_ball, ts):
    vx = (cur_ball[X] - prev_ball[X])/ts
    vy = (cur_ball[Y] - prev_ball[Y])/ts
    vd = math.atan2(vy, vx)/math.pi
    vr = np.clip(math.sqrt(math.pow(vx, 2) + math.pow(vy, 2)), -10, 10)/10
    return [vd, vr]

def predict_robot_velocity(cur_posture, prev_posture, index, ts):
    vx = (cur_posture[index][X] - prev_posture[index][X])/ts
    vy = (cur_posture[index][Y] - prev_posture[index][Y])/ts
    vd = math.atan2(vy, vx)/math.pi
    vr = np.clip(math.sqrt(math.pow(vx, 2) + math.pow(vy, 2)), -10, 10)/10
    return [vd, vr]


def get_global_state(cur_posture, prev_posture, cur_posture_opp, prev_posture_opp, cur_ball, prev_ball, field, goal, max_linear_velocity):
    ##### for state
    # ball_velocity = predict_ball_velocity(cur_ball, prev_ball, 0.05)
    # robot_velocity = [predict_robot_velocity(cur_posture, prev_posture, a, 0.05) for a in range(5)]
    # opp_robot_velocity = [predict_robot_velocity(cur_posture_opp, prev_posture_opp, a, 0.05) for a in range(5)]
    x0 = (goal[X] + field[X])/2
    y0 = 0
    pxx = field[X] + 2*goal[X]
    pyy = field[Y] 
    pdd = math.sqrt(pow(pxx,2) + pow(pyy,2))
    step_time = 0.05
    states =[ round(helper.distance(x0, cur_ball[X], y0, cur_ball[Y])/pdd, 2), round(math.atan2(cur_ball[Y] - y0, cur_ball[X] - x0)/math.pi, 2),
     round(helper.distance(x0, cur_posture[0][X], y0, cur_posture[0][Y])/pdd, 2), round(math.atan2(cur_posture[0][Y] - y0, cur_posture[0][X] - x0)/math.pi, 2), 
     round(helper.distance(x0, cur_posture[1][X], y0, cur_posture[1][Y])/pdd, 2), round(math.atan2(cur_posture[1][Y] - y0, cur_posture[1][X] - x0)/math.pi, 2), 
     round(helper.distance(x0, cur_posture[2][X], y0, cur_posture[2][Y])/pdd, 2), round(math.atan2(cur_posture[2][Y] - y0, cur_posture[2][X] - x0)/math.pi, 2), 
     round(helper.distance(x0, cur_posture[3][X], y0, cur_posture[3][Y])/pdd, 2), round(math.atan2(cur_posture[3][Y] - y0, cur_posture[3][X] - x0)/math.pi, 2), 
     round(helper.distance(x0, cur_posture[4][X], y0, cur_posture[4][Y])/pdd, 2), round(math.atan2(cur_posture[4][Y] - y0, cur_posture[4][X] - x0)/math.pi, 2), 
     round(helper.distance(x0, cur_posture_opp[0][X], y0, cur_posture_opp[0][Y])/pdd, 2), round(math.atan2(cur_posture_opp[0][Y] - y0, cur_posture_opp[0][X] - x0)/math.pi, 2), 
     round(helper.distance(x0, cur_posture_opp[1][X], y0, cur_posture_opp[1][Y])/pdd, 2), round(math.atan2(cur_posture_opp[1][Y] - y0, cur_posture_opp[1][X] - x0)/math.pi, 2), 
     round(helper.distance(x0, cur_posture_opp[2][X], y0, cur_posture_opp[2][Y])/pdd, 2), round(math.atan2(cur_posture_opp[2][Y] - y0, cur_posture_opp[2][X] - x0)/math.pi, 2), 
     round(helper.distance(x0, cur_posture_opp[3][X], y0, cur_posture_opp[3][Y])/pdd, 2), round(math.atan2(cur_posture_opp[3][Y] - y0, cur_posture_opp[3][X] - x0)/math.pi, 2), 
     round(helper.distance(x0, cur_posture_opp[4][X], y0, cur_posture_opp[4][Y])/pdd, 2), round(math.atan2(cur_posture_opp[4][Y] - y0, cur_posture_opp[4][X] - x0)/math.pi, 2) ]
    return states

def get_state(cur_posture, prev_posture, cur_posture_opp, prev_posture_opp, cur_ball, prev_ball, field, goal, max_linear_velocity):
    ##### for state
    states = [[] for _ in range(5)]
    x0 = (goal[X] + field[X])/2
    y0 = 0
    pxx = field[X] + 2*goal[X]
    pyy = field[Y] 
    pdd = math.sqrt(pow(pxx,2) + pow(pyy,2))
    obs_range = 2.0
    step_time = 0.05

    for i in range(5):

        ball_robot_dis = helper.distance(cur_posture[i][X] , cur_ball[X], cur_posture[i][Y], cur_ball[Y])
        ball_robot_dis_prev = helper.distance(prev_posture[i][X] , prev_ball[X], prev_posture[i][Y], prev_ball[Y])
        ball_robot_velocity = (ball_robot_dis_prev - ball_robot_dis)/step_time
        ball_robot_th_error = math.atan2(cur_ball[Y]-cur_posture[i][Y], cur_ball[X]-cur_posture[i][X]) - cur_posture[i][TH]

        ball_opp_goal = helper.distance((goal[X] + field[X])/2 , cur_ball[X], 0, cur_ball[Y])
        ball_opp_goal_th = math.atan2(cur_ball[Y] - y0, cur_ball[X] - x0) 
        ball_opp_goal_prev = helper.distance((goal[X] + field[X])/2 , prev_ball[X], 0, prev_ball[Y])
        ball_opp_goal_velocity = (ball_opp_goal_prev - ball_opp_goal)/step_time
        robot_opp_goal = helper.distance((goal[X] + field[X])/2 , cur_posture[i][X], 0, cur_posture[i][Y])
        robot_opp_goal_th = math.atan2(cur_posture[i][Y] - y0, cur_posture[i][X] - x0)

        GK_defense_angle = helper.get_defense_kick_angle(cur_ball, field, cur_ball)
        GK_defense_x = math.cos(GK_defense_angle)*0.6 - field[X]/2
        GK_defense_y = math.sin(GK_defense_angle)*0.6
        GK_th_error = math.atan2(GK_defense_y-cur_posture[i][Y], GK_defense_x-cur_posture[i][X]) - cur_posture[i][TH]
        GK_defense_dis = helper.distance(cur_posture[i][X], GK_defense_x, cur_posture[i][Y], GK_defense_y) 
        
        D12_defense_x = 0.5*(cur_ball[X] - (goal[X] + field[X])/2)
        D12_defense_y = 0.5*(cur_ball[Y])
        D12_th_error = math.atan2(D12_defense_y-cur_posture[i][Y], D12_defense_x-cur_posture[i][X]) - cur_posture[i][TH]
        D12_defense_dis = helper.distance(cur_posture[i][X] , D12_defense_x, cur_posture[i][Y], D12_defense_y)
        
        F12_attack_x = 0.5*(cur_ball[X] + field[X]/6) 
        F12_attack_y = 0.5*(cur_ball[Y])
        F12_th_error = math.atan2(F12_attack_y-cur_posture[i][Y], F12_attack_x-cur_posture[i][X]) - cur_posture[i][TH]
        F12_attack_dis = helper.distance(cur_posture[i][X] , F12_attack_x, cur_posture[i][Y], F12_attack_y)

        closest = 0
        if i != 0:
            idx = helper.find_closest_robot(cur_ball,cur_posture,5)
            if i == idx:
                closest = 1
                
                D12_defense_x = cur_ball[X]
                D12_defense_y = cur_ball[Y]
                D12_th_error = math.atan2(D12_defense_y-cur_posture[i][Y], D12_defense_x-cur_posture[i][X]) - cur_posture[i][TH]
                D12_defense_dis = helper.distance(cur_posture[i][X] , D12_defense_x, cur_posture[i][Y], D12_defense_y)
                
                F12_attack_x = cur_ball[X]
                F12_attack_y = cur_ball[Y]
                F12_th_error = math.atan2(F12_attack_y-cur_posture[i][Y], F12_attack_x-cur_posture[i][X]) - cur_posture[i][TH]
                F12_attack_dis = helper.distance(cur_posture[i][X] , F12_attack_x, cur_posture[i][Y], F12_attack_y)


        if i == 0:
            states[i] =[closest, int(cur_posture[i][BALL_POSSESSION]), round(GK_defense_dis/pdd, 2), round(ball_robot_dis/pdd, 2), round(np.clip(ball_robot_velocity, -10, 10)/10, 2), round(ball_robot_th_error/math.pi, 2),
                        round(cur_posture[i][TH]/math.pi, 2), round(robot_opp_goal/pdd, 2), round(robot_opp_goal_th/math.pi, 2), round(ball_opp_goal/pdd, 2), round(ball_opp_goal_th/math.pi, 2), round(np.clip(ball_opp_goal_velocity, -10, 10)/10, 2)]
        elif (i == 1) or (i == 2): 
            states[i] =[closest, int(cur_posture[i][BALL_POSSESSION]), round(D12_defense_dis/pdd, 2), round(ball_robot_dis/pdd, 2), round(np.clip(ball_robot_velocity, -10, 10)/10, 2), round(ball_robot_th_error/math.pi, 2),
                        round(cur_posture[i][TH]/math.pi, 2), round(robot_opp_goal/pdd, 2), round(robot_opp_goal_th/math.pi, 2), round(ball_opp_goal/pdd, 2), round(ball_opp_goal_th/math.pi, 2), round(np.clip(ball_opp_goal_velocity, -10, 10)/10, 2)]
        elif (i == 3) or (i == 4): 
            states[i] =[closest, int(cur_posture[i][BALL_POSSESSION]), round(F12_attack_dis/pdd, 2), round(ball_robot_dis/pdd, 2), round(np.clip(ball_robot_velocity, -10, 10)/10, 2), round(ball_robot_th_error/math.pi, 2),
                        round(cur_posture[i][TH]/math.pi, 2), round(robot_opp_goal/pdd, 2), round(robot_opp_goal_th/math.pi, 2), round(ball_opp_goal/pdd, 2), round(ball_opp_goal_th/math.pi, 2), round(np.clip(ball_opp_goal_velocity, -10, 10)/10, 2)]

        for j in range(5):
            if (helper.distance(cur_posture[j][X], cur_posture[i][X], cur_posture[j][Y], cur_posture[i][Y]) > obs_range) or (j == i ):
                states[i].append(0)
                states[i].append(0)
            else:
                states[i].append(round(helper.distance(cur_posture[j][X], cur_posture[i][X], cur_posture[j][Y], cur_posture[i][Y])/obs_range, 2))
                states[i].append(round((math.atan2(cur_posture[j][Y]-cur_posture[i][Y], cur_posture[j][X]-cur_posture[i][X]) - cur_posture[i][TH])/math.pi, 2))
        for j in range(5):
            if helper.distance(cur_posture_opp[j][X], cur_posture[i][X], cur_posture_opp[j][Y], cur_posture[i][Y]) > obs_range:
                states[i].append(0)
                states[i].append(0)
            else:
                states[i].append(round(helper.distance(cur_posture_opp[j][X], cur_posture[i][X], cur_posture_opp[j][Y], cur_posture[i][Y])/obs_range, 2))
                states[i].append(round((math.atan2(cur_posture_opp[j][Y]-cur_posture[i][Y], cur_posture_opp[j][X]-cur_posture[i][X]) - cur_posture[i][TH])/math.pi, 2))


    return states
 
def get_reward(cur_posture, prev_posture, cur_ball, prev_ball, field, goal, id, touch_counter, reset_reason):
    step_time = 0.05
    ball_robot_dis = helper.distance(cur_posture[id][X] , cur_ball[X], cur_posture[id][Y], cur_ball[Y])
    ball_robot_dis_prev = helper.distance(prev_posture[id][X] , prev_ball[X], prev_posture[id][Y], prev_ball[Y])
    ball_robot_velocity = (ball_robot_dis_prev - ball_robot_dis)/step_time
    ball_opp_goal = helper.distance((goal[X] + field[X])/2 , cur_ball[X], 0, cur_ball[Y])
    ball_opp_goal_prev = helper.distance((goal[X] + field[X])/2 , prev_ball[X], 0, prev_ball[Y])
    ball_opp_goal_velocity = (ball_opp_goal_prev - ball_opp_goal)/step_time
    # defense_angle_prev = helper.get_defense_kick_angle(prev_ball, field, prev_ball)
    # defense_x_prev = math.cos(defense_angle_prev)*0.6 - field[X]/2
    # defense_y_prev = math.sin(defense_angle_prev)*0.6
    # defense_dis_prev = helper.distance(prev_posture[id][X], defense_x_prev, prev_posture[id][Y], defense_y_prev) 
    # defense_position_velocity = (defense_dis_prev - defense_dis)/step_time 
    ball_robot_th_error = abs(math.atan2(cur_ball[Y]-cur_posture[id][Y], cur_ball[X]-cur_posture[id][X]) - cur_posture[id][TH])
    GK_defense_angle = helper.get_defense_kick_angle(cur_ball, field, cur_ball)
    GK_defense_x = math.cos(GK_defense_angle)*0.6 - field[X]/2
    GK_defense_y = math.sin(GK_defense_angle)*0.6
    GK_defense_dis = helper.distance(cur_posture[id][X], GK_defense_x, cur_posture[id][Y], GK_defense_y) 
    GK_th_error = abs(math.atan2(GK_defense_y-cur_posture[id][Y], GK_defense_x-cur_posture[id][X]) - cur_posture[id][TH])
    D12_defense_x = 0.5*(cur_ball[X] - (goal[X] + field[X])/2)
    D12_defense_y = 0.5*(cur_ball[Y])
    D12_defense_dis = helper.distance(cur_posture[id][X] , D12_defense_x, cur_posture[id][Y], D12_defense_y)
    D12_th_error = abs(math.atan2(D12_defense_y-cur_posture[id][Y], D12_defense_x-cur_posture[id][X]) - cur_posture[id][TH])
    F12_attack_x = (cur_ball[X]) 
    F12_attack_y = (cur_ball[Y])
    F12_attack_dis = helper.distance(cur_posture[id][X] , F12_attack_x, cur_posture[id][Y], F12_attack_y)
    F12_th_error = abs(math.atan2(F12_attack_y-cur_posture[id][Y], F12_attack_x-cur_posture[id][X]) - cur_posture[id][TH])
    
    i = id
    if i != 0:
        j = i + 1
        if i % 2 == 0:
            j = i - 1
        c_ball_robot_dis = helper.distance(cur_posture[j][X] , cur_ball[X], cur_posture[j][Y], cur_ball[Y])
        closest = int(ball_robot_dis < c_ball_robot_dis)
        if closest:
            D12_defense_x = cur_ball[X]
            D12_defense_y = cur_ball[Y]
            D12_th_error = math.atan2(D12_defense_y-cur_posture[i][Y], D12_defense_x-cur_posture[i][X]) - cur_posture[i][TH]
            D12_defense_dis = helper.distance(cur_posture[i][X] , D12_defense_x, cur_posture[i][Y], D12_defense_y)
            
            F12_attack_x = cur_ball[X]
            F12_attack_y = cur_ball[Y]
            F12_th_error = math.atan2(F12_attack_y-cur_posture[i][Y], F12_attack_x-cur_posture[i][X]) - cur_posture[i][TH]
            F12_attack_dis = helper.distance(cur_posture[i][X] , F12_attack_x, cur_posture[i][Y], F12_attack_y)

    isTouchedMove = touch_counter.isTouchedMoveCurrent()
    differential = 0
    delta_ball2goal = ball_opp_goal - ball_opp_goal_prev
    if isTouchedMove[id] > 0:
        if delta_ball2goal <= 0:
            differential = differential + ( (-50) * delta_ball2goal )
    try:
        if id == 0:
            return  ( 1.0*(math.exp(-1*GK_defense_dis)) + 0.5*(math.exp(-1*ball_robot_th_error)) + 0.5*np.clip(1-math.exp(-1*(ball_robot_velocity)),0,1) + differential)
        elif (id == 1) or (id == 2): 
            return  ( 1.0*(math.exp(-1*D12_defense_dis)) + 0.5*(math.exp(-1*ball_robot_th_error)) + 0.5*np.clip(1-math.exp(-1*(ball_robot_velocity)),0,1) + differential) 
        elif (id == 3) or (id == 4): 
            return  ( 1.0*(math.exp(-1*F12_attack_dis)) + 0.5*(math.exp(-1*ball_robot_th_error)) + 0.5*np.clip(1-math.exp(-1*(ball_robot_velocity)),0,1) + differential) 
    except:    
        return 0
        pass

    # if id == 0:
    #     return  ( 0.5*(math.exp(-1*defense_dis)) + 0.5*(math.exp(-1*gk_th_error)) + np.clip(1-math.exp(-1*(defense_position_velocity)),0,1) )
    # else: 
    #     return  ( 0.5*(math.exp(-1*ball_robot_dis)) + 0.5*(math.exp(-1*robot_th_error)) + np.clip(1-math.exp(-1*(ball_robot_velocity)),0,1) ) 

def get_team_reward(cur_posture, prev_posture, cur_ball, prev_ball, field, goal):
    step_time = 0.05
    ball_opp_goal = helper.distance((goal[X] + field[X])/2 , cur_ball[X], 0, cur_ball[Y])
    ball_opp_goal_prev = helper.distance((goal[X] + field[X])/2 , prev_ball[X], 0, prev_ball[Y])
    ball_opp_goal_velocity = (ball_opp_goal_prev - ball_opp_goal)/step_time
    try:
        
        if cur_ball[X] > field[X]/2 :
            score = 100
        elif cur_ball[X] < - field[X]/2 :
            score = 0
        else:
            score = 0

        return  ( 5*(math.exp(-1*(ball_opp_goal))) + 5*np.clip(1-math.exp(-1*(ball_opp_goal_velocity)),0,1) + score)
    except:    
        return 0
        pass

    # return  ( (math.exp(-1*(ball_opp_goal**2))-1) + np.clip(1-math.exp(-1*(ball_opp_goal_velocity**2)),0,1) + score)



def get_action(robot_id, action_number, max_linear_velocity):
    # 20 actions: go forwards, go backwards, rotate right, rotate left, stop
    max_vel = max_linear_velocity
    GK_WHEELS = [  
            [  1.00*max_vel[0],   1.00*max_vel[0],  0.0,  0.0,  0.0, 1], # go forwards
            [ -0.70*max_vel[0],  -0.70*max_vel[0],  0.0,  0.0,  0.0, 1], # go backwards
            [  1.00*max_vel[0],   0.50*max_vel[0],  0.0,  0.0,  0.0, 1], # forwards turn
            [  0.50*max_vel[0],   1.00*max_vel[0],  0.0,  0.0,  0.0, 1], # forwards turn
            [  0.65*max_vel[0],   0.20*max_vel[0],  0.0,  0.0,  0.0, 1], # forwards turn
            [  0.20*max_vel[0],   0.65*max_vel[0],  0.0,  0.0,  0.0, 1], # forwards turn
            [  0.50*max_vel[0],   0.10*max_vel[0],  0.0,  0.0,  0.0, 1], # forwards turn
            [  0.10*max_vel[0],   0.50*max_vel[0],  0.0,  0.0,  0.0, 1], # forwards turn
            [ -0.65*max_vel[0],  -0.20*max_vel[0],  0.0,  0.0,  0.0, 1], # backwards turn
            [ -0.20*max_vel[0],  -0.65*max_vel[0],  0.0,  0.0,  0.0, 1], # backwards turn
            [ -1.00*max_vel[0],  -0.50*max_vel[0],  0.0,  0.0,  0.0, 1], # backwards turn
            [ -0.50*max_vel[0],  -1.00*max_vel[0],  0.0,  0.0,  0.0, 1], # backwards turn
            [  0.23*max_vel[0],  -0.23*max_vel[0],  0.0,  0.0,  0.0, 1], # turn
            [ -0.23*max_vel[0],   0.23*max_vel[0],  0.0,  0.0,  0.0, 1], # turn
            [  1.00*max_vel[0],   0.85*max_vel[0],  2.0,  2.0,  0.0, 1], # forwards turn and kick
            [  0.85*max_vel[0],   1.00*max_vel[0],  2.0,  2.0,  0.0, 1], # forwards turn and kick
            [  0.50*max_vel[0],   0.50*max_vel[0],  5.0,  0.0,  0.0, 1], # forwards kick
            [  0.50*max_vel[0],   0.50*max_vel[0],  8.0,  8.0,  0.0, 1], # forwards kick
            [  0.00*max_vel[0],   0.00*max_vel[0],  8.0,  8.0,  0.0, 1], # stop kick
            [  0.00*max_vel[0],   0.00*max_vel[0],  0.0,  0.0,  0.0, 1], # stop
            ]

    D12_WHEELS = [  
            [  1.00*max_vel[2],   1.00*max_vel[2],  0.0,  0.0,  0.0, 1], # go forwards
            [ -0.70*max_vel[2],  -0.70*max_vel[2],  0.0,  0.0,  0.0, 1], # go backwards
            [  1.00*max_vel[2],   0.50*max_vel[2],  0.0,  0.0,  0.0, 1], # forwards turn
            [  0.50*max_vel[2],   1.00*max_vel[2],  0.0,  0.0,  0.0, 1], # forwards turn
            [  0.65*max_vel[2],   0.20*max_vel[2],  0.0,  0.0,  0.0, 1], # forwards turn
            [  0.20*max_vel[2],   0.65*max_vel[2],  0.0,  0.0,  0.0, 1], # forwards turn
            [  0.50*max_vel[2],   0.10*max_vel[2],  0.0,  0.0,  0.0, 1], # forwards turn
            [  0.10*max_vel[2],   0.50*max_vel[2],  0.0,  0.0,  0.0, 1], # forwards turn
            [ -0.65*max_vel[2],  -0.20*max_vel[2],  0.0,  0.0,  0.0, 1], # backwards turn
            [ -0.20*max_vel[2],  -0.65*max_vel[2],  0.0,  0.0,  0.0, 1], # backwards turn
            [ -1.00*max_vel[2],  -0.50*max_vel[2],  0.0,  0.0,  0.0, 1], # backwards turn
            [ -0.50*max_vel[2],  -1.00*max_vel[2],  0.0,  0.0,  0.0, 1], # backwards turn
            [  0.23*max_vel[2],  -0.23*max_vel[2],  0.0,  0.0,  0.0, 1], # turn
            [ -0.23*max_vel[2],   0.23*max_vel[2],  0.0,  0.0,  0.0, 1], # turn
            [  1.00*max_vel[2],   0.85*max_vel[2],  2.0,  2.0,  0.0, 1], # forwards turn and kick
            [  0.85*max_vel[2],   1.00*max_vel[2],  2.0,  2.0,  0.0, 1], # forwards turn and kick
            [  0.50*max_vel[2],   0.50*max_vel[2],  5.0,  0.0,  0.0, 1], # forwards kick
            [  0.50*max_vel[2],   0.50*max_vel[2],  8.0,  8.0,  0.0, 1], # forwards kick
            [  0.00*max_vel[2],   0.00*max_vel[2],  8.0,  8.0,  0.0, 1], # stop kick
            [  0.00*max_vel[2],   0.00*max_vel[2],  0.0,  0.0,  0.0, 1], # stop
            ]


    F12_WHEELS = [  
            [  1.00*max_vel[4],   1.00*max_vel[4],  0.0,  0.0,  0.0, 1], # go forwards
            [ -0.70*max_vel[4],  -0.70*max_vel[4],  0.0,  0.0,  0.0, 1], # go backwards
            [  1.00*max_vel[4],   0.50*max_vel[4],  0.0,  0.0,  0.0, 1], # forwards turn
            [  0.50*max_vel[4],   1.00*max_vel[4],  0.0,  0.0,  0.0, 1], # forwards turn
            [  0.65*max_vel[4],   0.20*max_vel[4],  0.0,  0.0,  0.0, 1], # forwards turn
            [  0.20*max_vel[4],   0.65*max_vel[4],  0.0,  0.0,  0.0, 1], # forwards turn
            [  0.50*max_vel[4],   0.10*max_vel[4],  0.0,  0.0,  0.0, 1], # forwards turn
            [  0.10*max_vel[4],   0.50*max_vel[4],  0.0,  0.0,  0.0, 1], # forwards turn
            [ -0.65*max_vel[4],  -0.20*max_vel[4],  0.0,  0.0,  0.0, 1], # backwards turn
            [ -0.20*max_vel[4],  -0.65*max_vel[4],  0.0,  0.0,  0.0, 1], # backwards turn
            [ -1.00*max_vel[4],  -0.50*max_vel[4],  0.0,  0.0,  0.0, 1], # backwards turn
            [ -0.50*max_vel[4],  -1.00*max_vel[4],  0.0,  0.0,  0.0, 1], # backwards turn
            [  0.23*max_vel[4],  -0.23*max_vel[4],  0.0,  0.0,  0.0, 1], # turn
            [ -0.23*max_vel[4],   0.23*max_vel[4],  0.0,  0.0,  0.0, 1], # turn
            [  1.00*max_vel[4],   0.85*max_vel[4],  2.0,  2.0,  0.0, 1], # forwards turn and kick
            [  0.85*max_vel[4],   1.00*max_vel[4],  2.0,  2.0,  0.0, 1], # forwards turn and kick
            [  0.50*max_vel[4],   0.50*max_vel[4],  5.0,  0.0,  0.0, 1], # forwards kick
            [  0.50*max_vel[4],   0.50*max_vel[4],  8.0,  8.0,  0.0, 1], # forwards kick
            [  0.00*max_vel[4],   0.00*max_vel[4],  8.0,  8.0,  0.0, 1], # stop kick
            [  0.00*max_vel[4],   0.00*max_vel[4],  0.0,  0.0,  0.0, 1], # stop
            ]

                
    wheel = [GK_WHEELS, D12_WHEELS, D12_WHEELS, F12_WHEELS, F12_WHEELS]


    return wheel[robot_id][action_number]



def get_hybrid_action(ActionControl, robot_id, action_number, ball_possesion):
    # 20 actions: go forwards, go backwards, rotate right, rotate left, stop

    if not ball_possesion:
        action_list = [
            ActionControl.go_to_ball,
            ActionControl.go_to_predicted_ball,
            ActionControl.go_to_predicted_ball2,
            ActionControl.turn_to_ball,
            ActionControl.go_to_position,
            # ActionControl.go_to_position2,
            # ActionControl.go_to_position3,
            ActionControl.stop,
        ]
    else:
        action_list = [
            ActionControl.go_to_goal,
            ActionControl.go_to_goal_l,
            ActionControl.go_to_goal_r,
            # ActionControl.turn_to_goal,
            ActionControl.pass_to_robot,
            ActionControl.cross_to_robot,
            ActionControl.shoot,
            # ActionControl.stop,
        ]

    wheel = action_list[action_number](robot_id)


    return wheel


class Logger():
    def __init__(self):

        self.episode = []
        self.m_episode = []
        self.num_value = 5
        self.value = [[] for _ in range(self.num_value)]
        self.mean_value =  [[] for _ in range(self.num_value)]

    def update(self, episode, value, num):



        self.episode.append(episode)
        for role in range(self.num_value):
            self.value[role].append(value[role])
        self.num = num
        if len(self.value[0]) >= self.num :
            self.m_episode.append(episode - self.num/2)
            for role in range(self.num_value):
                self.mean_value[role].append(np.mean(self.value[role][-self.num:]))

    def plot(self, name):
        plt.title(str(name))
        xlab = 'Training Episodes'
        ylab = 'Role and Team Reward'
        color=['b','g','c','y','r']
        label=['GK_Reward','D12_Reward','F12_Reward','Team_Reward','Total_Reward']
        for role in range(self.num_value):
            # plt.plot(self.episode, self.value, c = 'lightskyblue', label='total_reward') 
            plt.xticks([ 20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000, 180000, 200000, 220000, 240000, 260000, 280000, 300000, 320000, 340000, 360000, 380000, 400000],
             ['20k', '40k', '60k', '80k', '100k', '120k', '140k', '160k', '180k', '200k', '220k', '240k', '260k', '280k', '300k', '320k', '340k', '360k', '380k', '400k'])
            plt.xlabel(xlab)
            plt.ylabel(ylab)
            plt.plot(self.m_episode[1:], self.mean_value[role][1:], c = color[role], label= label[role]) 
        plt.legend(loc=2)
        fig_path = os.path.join(os.path.dirname(__file__), 'TOTAL_'+str(name)+'.png')
        plt.grid(True)
        plt.savefig(fig_path)
        data_path = os.path.join(os.path.dirname(__file__), str(name)+'_data.pickle')
        with open(data_path,"wb") as fw:
            pickle.dump(self.value,fw)
        plt.close()

  

