#!/usr/bin/env python3

# Author(s): Taeyoung Kim, Chansol Hong, Luiz Felipe Vecchietti
# Maintainer: Chansol Hong (cshong@rit.kaist.ac.kr)

import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../common')
try:
    from participant import Game, Frame
except ImportError as err:
    print('player_rulebasedC: \'participant\' module cannot be imported:', err)
    raise

import math
import helper

from scipy.optimize import fsolve
import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')

#reset_reason
NONE = Game.NONE
GAME_START = Game.GAME_START
SCORE_MYTEAM = Game.SCORE_MYTEAM
SCORE_OPPONENT = Game.SCORE_OPPONENT
GAME_END = Game.GAME_END
DEADLOCK = Game.DEADLOCK
GOALKICK = Game.GOALKICK
CORNERKICK = Game.CORNERKICK
PENALTYKICK = Game.PENALTYKICK
HALFTIME = Game.HALFTIME
EPISODE_END = Game.EPISODE_END

#game_state
STATE_DEFAULT = Game.STATE_DEFAULT
STATE_KICKOFF = Game.STATE_KICKOFF
STATE_GOALKICK = Game.STATE_GOALKICK
STATE_CORNERKICK = Game.STATE_CORNERKICK
STATE_PENALTYKICK = Game.STATE_PENALTYKICK

#coordinates
MY_TEAM = Frame.MY_TEAM
OP_TEAM = Frame.OP_TEAM
BALL = Frame.BALL
X = Frame.X
Y = Frame.Y
Z = Frame.Z
TH = Frame.TH
ACTIVE = Frame.ACTIVE
TOUCH = Frame.TOUCH
BALL_POSSESSION = Frame.BALL_POSSESSION

class ActionControl:

    def __init__(self, max_linear_velocity, field, goal):
        self.max_linear_velocity = max_linear_velocity
        self.field = field
        self.goal = goal
        self.g = 9.81 # gravity
        self.damping = 0.2 # linear damping
        self.mult_fs = 0.75 
        self.max_kick_speed = 10*self.mult_fs # 7.5 m/s
        self.mult_angle = 5
        self.max_kick_angle = 10*self.mult_angle # 50 degrees

        self.cur_posture = []
        self.prev_posture = []
        self.cur_posture_opp = []
        self.prev_posture_opp = []
        self.cur_ball = []
        self.prev_ball = []
        self.reset_reason = NONE

    def update_state(self, cur_posture, prev_posture, cur_ball, prev_ball, reset_reason):
        self.cur_posture = cur_posture
        self.prev_posture = prev_posture
        self.cur_ball = cur_ball
        self.prev_ball = prev_ball
        self.reset_reason = reset_reason

        self.predicted_ball = helper.predict_ball(self.cur_ball, self.prev_ball, self.reset_reason)
        self.defense_angle = helper.get_defense_kick_angle(self.predicted_ball, self.field, self.cur_ball)
        self.attack_angle = helper.get_attack_kick_angle(self.predicted_ball, self.field)


    def go_to(self, robot_id, x, y):
        sign = 1
        kd = 7 if ((robot_id == 1) or (robot_id == 2)) else 5
        ka = 0.3

        tod = 0.005 # tolerance of distance
        tot = math.pi/360 # tolerance of theta

        dx = x - self.cur_posture[robot_id][X]
        dy = y - self.cur_posture[robot_id][Y]
        d_e = math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))
        desired_th = math.atan2(dy, dx)

        d_th = helper.wrap_to_pi(desired_th - self.cur_posture[robot_id][TH])
        
        if (d_th > helper.degree2radian(90)):
            d_th -= math.pi
            sign = -1
        elif (d_th < helper.degree2radian(-90)):
            d_th += math.pi
            sign = -1

        if (d_e < tod):
            kd = 0
        if (abs(d_th) < tot):
            ka = 0

        if self.go_fast(robot_id):
            kd *= 5

        left_wheel, right_wheel = helper.set_wheel_velocity(self.max_linear_velocity[robot_id],
                    sign * (kd * d_e - ka * d_th), 
                    sign * (kd * d_e + ka * d_th))

        return [left_wheel, right_wheel, 0, 0, 0, 1]

    def go_fast(self, robot_id):
        distance2ball = helper.distance(self.cur_ball[X], self.cur_posture[robot_id][X],
                                    self.cur_ball[Y], self.cur_posture[robot_id][Y])
        d_bg = helper.distance(self.cur_ball[X], 3.9,
                                    self.cur_ball[Y], 0)
        d_rg = helper.distance(3.9, self.cur_posture[robot_id][X],
                                    0, self.cur_posture[robot_id][Y])
        
        if (distance2ball < 0.25 and d_rg > d_bg):
            if (self.cur_ball[X] > 3.7 and abs(self.cur_ball[Y]) > 0.5 and abs(self.cur_posture[robot_id][TH]) < 30 * math.pi/180):
                return False
            else:
                return True
        else:
            return False

    def turn_to(self, robot_id, x, y):
        ka = 0.2
        tot = math.pi/360

        dx = x - self.cur_posture[robot_id][X]
        dy = y - self.cur_posture[robot_id][Y]
        desired_th = math.atan2(dy, dx)
        d_th = helper.wrap_to_pi(desired_th - self.cur_posture[robot_id][TH])
        
        if (abs(d_th) < tot):
            ka = 0
        
        left_wheel, right_wheel = helper.set_wheel_velocity(self.max_linear_velocity[robot_id],
                                                                -ka*d_th,
                                                                ka*d_th)

        return [left_wheel, right_wheel, 0, 0, 0 , 1]

    def defend_ball(self, robot_id):
        if robot_id != 0:
            return None
        if self.reset_reason != NONE:
            return None

        # GK takes 250ms to perform defense move
        dx = self.cur_ball[X] - self.prev_ball[X]
        dy = self.cur_ball[Y] - self.prev_ball[Y]
        dz = self.cur_ball[Z] - self.prev_ball[Z]
        predicted_ball_gk = [self.cur_ball[X] + 5*dx, self.cur_ball[Y] + 5*dy, self.cur_ball[Z] + 5*dz]

        if predicted_ball_gk[X] < self.cur_posture[robot_id][X] + 0.1:
            # right part of the goal
            if -0.65 < predicted_ball_gk[Y] < -0.07:
                # top part of the goal
                if (predicted_ball_gk[Z] > 0.25):
                    return [0, 0, 0, 0, 7, 0]
                else:
                    return [0, 0, 0, 0, 6, 0]
            # center part of the goal
            if -0.07 < predicted_ball_gk[Y] < 0.07:
                # top part of the goal
                if (predicted_ball_gk[Z] > 0.25):
                    return [0, 0, 0, 0, 8, 0]
                else:
                    return [0, 0, 0, 0, 3, 0]
            # left part of the goal
            if 0.07 < predicted_ball_gk[Y] < 0.65:
                # top part of the goal
                if (predicted_ball_gk[Z] > 0.25):
                    return [0, 0, 0, 0, 9, 0]
                else:
                    return [0, 0, 0, 0, 10, 0]
        else:
            return None

    def pass_to(self, robot_id, x, y):
        
        dist = helper.distance(self.cur_posture[robot_id][X], x, self.cur_posture[robot_id][Y], y)
        kick_speed = (7 + 1.5 * (dist / 5.07))*self.mult_fs
        kick_angle = 0

        direction = math.atan2(y - self.cur_posture[robot_id][Y], x - self.cur_posture[robot_id][X]) * 4 / math.pi
        if direction > 4:
            direction -= 8

        if abs(self.cur_posture[robot_id][TH] - math.pi / 4 * direction) > math.pi:
            if 0 <= abs(2 * math.pi + self.cur_posture[robot_id][TH] - math.pi / 4 * direction) <= math.pi:
                self.cur_posture[robot_id][TH] += 2 * math.pi
            else:
                self.cur_posture[robot_id][TH] -= 2 * math.pi

        if self.cur_posture[robot_id][TH] > math.pi / 4 * direction:
            if self.cur_posture[robot_id][TH] * 180 / math.pi - 45 * direction > 5:
                w = min(1, (self.cur_posture[robot_id][TH] * 180 / math.pi - 45 * direction) / (70 * self.max_linear_velocity[robot_id] / self.max_linear_velocity[0]))
                if helper.distance(self.cur_posture[robot_id][X], self.prev_posture[robot_id][X], self.cur_posture[robot_id][Y], self.prev_posture[robot_id][Y]) < 0.01: # corner case
                    return [w/2, -w/2, 0, 0, 0, 1]
                return [0.4 + w/2, 0.4 - w/2, 0, 0, 0, 1]
            else:
                return [1, 1, kick_speed, kick_angle, 0, 1]
        else:
            if self.cur_posture[robot_id][TH] * 180 / math.pi - 45 * direction < -5:
                w = min(1, -(self.cur_posture[robot_id][TH] * 180 / math.pi - 45 * direction) / (70 * self.max_linear_velocity[robot_id] / self.max_linear_velocity[0]))
                if helper.distance(self.cur_posture[robot_id][X], self.prev_posture[robot_id][X], self.cur_posture[robot_id][Y], self.prev_posture[robot_id][Y]) < 0.01:
                    return [-w/2, w/2, 0, 0, 0, 1]
                return [0.4 - w/2, 0.4 + w/2, 0, 0, 0, 1]
            else:
                return [1, 1, kick_speed, kick_angle, 0, 1]

    def cross_to(self, robot_id, x, y, z):
        
        dist = helper.distance(self.cur_posture[robot_id][X], x, self.cur_posture[robot_id][Y], y)
        max_cross_angle = 40

        if self.damping == 0:
            try:
                theta = math.pi*max_cross_angle/180
                v0 = math.sqrt((self.g * dist * dist) / (2 * (math.cos(theta) ** 2) * (dist * math.tan(theta) - z)))
                while v0 > self.max_kick_speed:
                    theta -= math.pi / 180
                    v0 = math.sqrt((self.g * dist * dist) / (2 * (math.cos(theta) ** 2) * (dist * math.tan(theta) - z)))
            except ValueError as e:
                #helper.printConsole(e)
                return None
        else:
            try:
                theta = math.pi*max_cross_angle/180
                while True:
                    relative_height_for_time = lambda t: (-self.g * t / self.damping) + self.g * (1 - math.exp(-self.damping*t)) / (self.damping**2) + dist * math.tan(theta) - (z - self.cur_ball[Z])
                    t = float(fsolve(relative_height_for_time, 2))
                    vx0 = dist * self.damping / (1 - math.exp(-self.damping * t))
                    vy0 = vx0 * math.tan(theta)
                    v0 = math.sqrt(vx0 ** 2 + vy0 ** 2)
                    if v0 > self.max_kick_speed:
                        theta -= math.pi / 180
                        if theta < 0:
                            return None
                        continue
                    break
            except ValueError as e:
                #helper.printConsole(e)
                return None

        kick_speed = v0 / self.mult_fs
        kick_angle = theta * (180 / math.pi) / self.mult_angle

        direction = math.atan2(y - self.cur_posture[robot_id][Y], x - self.cur_posture[robot_id][X]) * 4 / math.pi
        if direction > 4:
            direction -= 8

        if abs(self.cur_posture[robot_id][TH] - math.pi / 4 * direction) > math.pi:
            if 0 <= abs(2 * math.pi + self.cur_posture[robot_id][TH] - math.pi / 4 * direction) <= math.pi:
                self.cur_posture[robot_id][TH] += 2 * math.pi
            else:
                self.cur_posture[robot_id][TH] -= 2 * math.pi

        if self.cur_posture[robot_id][TH] > math.pi / 4 * direction:
            if self.cur_posture[robot_id][TH] * 180 / math.pi - 45 * direction > 5:
                w = min(1, (self.cur_posture[robot_id][TH] * 180 / math.pi - 45 * direction) / (70 * self.max_linear_velocity[robot_id] / self.max_linear_velocity[0]))
                return [w/2, -w/2, 0, 0, 0, 1]
            else:
                if kick_angle > 10:
                    return [-1, -1, 0, 0, 0, 1]
                elif kick_speed > 10:
                    return [1, 1, 0, 0, 0, 1]
                else:
                    return [1, 1, kick_speed, kick_angle, 0, 1]
        else:
            if self.cur_posture[robot_id][TH] * 180 / math.pi - 45 * direction < -5:
                w = min(1, -(self.cur_posture[robot_id][TH] * 180 / math.pi - 45 * direction) / (70 * self.max_linear_velocity[robot_id] / self.max_linear_velocity[0]))
                return [-w/2, w/2, 0, 0, 0, 1]
            else:
                if kick_speed > 10:
                    return [1, 1, 0, 0, 0, 1]
                elif kick_angle > 10:
                    return [-1, -1, 0, 0, 0, 1]
                else:
                    return [1, 1, kick_speed, kick_angle, 0, 1]

    def shoot_to(self, robot_id, x, y, kick_speed=10, kick_angle=4):

        direction = math.atan2(y - self.cur_posture[robot_id][Y], x - self.cur_posture[robot_id][X]) * 4 / math.pi

        if direction > 4:
            direction -= 8

        if abs(self.cur_posture[robot_id][TH] - math.pi / 4 * direction) > math.pi:
            if 0 <= abs(2 * math.pi + self.cur_posture[robot_id][TH] - math.pi / 4 * direction) <= math.pi:
                self.cur_posture[robot_id][TH] += 2 * math.pi
            else:
                self.cur_posture[robot_id][TH] -= 2 * math.pi

        if self.cur_posture[robot_id][TH] > math.pi / 4 * direction:
            if self.cur_posture[robot_id][TH] * 180 / math.pi - 45 * direction < 15:
                return [1, 1, kick_speed, kick_angle, 0, 1]
            else:
                w = min(1, (self.cur_posture[robot_id][TH] * 180 / math.pi - 45 * direction) / (70 * self.max_linear_velocity[robot_id] / self.max_linear_velocity[0]))
                if helper.distance(self.cur_posture[robot_id][X], self.prev_posture[robot_id][X], self.cur_posture[robot_id][Y], self.prev_posture[robot_id][Y]) < 0.01: # corner case
                    return [w/2, -w/2, 0, 0, 0, 1]
                return [0.75 + w/2, 0.75 - w/2, 0, 0, 0, 1]
        else:
            if self.cur_posture[robot_id][TH] * 180 / math.pi - 45 * direction > -15:
                return [1, 1, kick_speed, kick_angle, 0, 1]
            else:
                w = min(1, -(self.cur_posture[robot_id][TH] * 180 / math.pi - 45 * direction) / (70 * self.max_linear_velocity[robot_id] / self.max_linear_velocity[0]))
                if helper.distance(self.cur_posture[robot_id][X], self.prev_posture[robot_id][X], self.cur_posture[robot_id][Y], self.prev_posture[robot_id][Y]) < 0.01:
                    return [-w/2, w/2, 0, 0, 0, 1]
                return [0.75 - w/2, 0.75 + w/2, 0, 0, 0, 1]



    def stop(self, robot_id):
        return [0,0,0,0,0,0]

    def go_to_ball(self, robot_id):
        x, y = self.cur_ball[:2]

        return self.go_to(robot_id, x, y)

    def go_to_predicted_ball(self, robot_id):
        x, y = helper.predict_ball(self.cur_ball, self.prev_ball, 4)

        return self.go_to(robot_id, x, y)

    def go_to_predicted_ball2(self, robot_id):
        x, y = helper.predict_ball(self.cur_ball, self.prev_ball, 10)

        return self.go_to(robot_id, x, y)

    def go_to_goal(self, robot_id):
        x, y = self.field[X]/2, 0

        return self.go_to(robot_id, x, y)

    def go_to_goal_l(self, robot_id):
        x, y = self.field[X]/2, -0.5

        return self.go_to(robot_id, x, y)

    def go_to_goal_r(self, robot_id):
        x, y = self.field[X]/2, 0.5

        return self.go_to(robot_id, x, y)

    def turn_to_ball(self, robot_id):
        x, y = self.cur_ball[:2]

        return self.turn_to(robot_id, x, y)

    def turn_to_goal(self, robot_id):
        x, y = self.field[X]/2, 0

        return self.turn_to(robot_id, x, y)

    def go_to_position(self, robot_id):
        GK_defense_x = math.cos(self.defense_angle)*0.6 - self.field[X]/2
        GK_defense_y = math.sin(self.defense_angle)*0.6
        D12_defense_x = 0.5*(self.cur_ball[X] - (self.goal[X] + self.field[X])/2)
        D12_defense_y = 0.5*(self.cur_ball[Y])
        F12_attack_x = 0.5*(self.cur_ball[X] + self.field[X]/6) 
        F12_attack_y = 0.5*(self.cur_ball[Y])
        i = robot_id
        if i != 0:
            idx = helper.find_closest_robot(self.cur_ball,self.cur_posture,5)
            if i == idx:
                D12_defense_x = self.cur_ball[X]
                D12_defense_y = self.cur_ball[Y]
                F12_attack_x = self.cur_ball[X]
                F12_attack_y = self.cur_ball[Y]

        dp = [
            [GK_defense_x, GK_defense_y],
            [D12_defense_x, D12_defense_y],
            [D12_defense_x, D12_defense_y],
            [F12_attack_x, F12_attack_y],
            [F12_attack_x, F12_attack_y],
        ]
        x, y = dp[robot_id]

        return self.go_to(robot_id, x, y)

    def go_to_position2(self, robot_id):
        GK_defense_x = math.cos(self.defense_angle)*0.9 - self.field[X]/2
        GK_defense_y = math.sin(self.defense_angle)*0.9
        D12_defense_x = 0.25*(self.cur_ball[X] - (self.goal[X] + self.field[X])/2)
        D12_defense_y = 0.25*(self.cur_ball[Y])
        F12_attack_x = 0.25*(self.cur_ball[X] + self.field[X]/6) 
        F12_attack_y = 0.25*(self.cur_ball[Y])
        i = robot_id
        if i != 0:
            idx = helper.find_closest_robot(self.cur_ball,self.cur_posture,5)
            if i == idx:
                D12_defense_x = 0.5*(self.cur_ball[X]+self.cur_posture[i][X])
                D12_defense_y = 0.5*(self.cur_ball[Y]+self.cur_posture[i][Y])
                F12_attack_x = 0.5*(self.cur_ball[X]+self.cur_posture[i][X])
                F12_attack_y = 0.5*(self.cur_ball[Y]+self.cur_posture[i][Y])
        dp = [
            [GK_defense_x, GK_defense_y],
            [D12_defense_x, D12_defense_y],
            [D12_defense_x, D12_defense_y],
            [F12_attack_x, F12_attack_y],
            [F12_attack_x, F12_attack_y],
        ]
        x, y = dp[robot_id]

        return self.go_to(robot_id, x, y)

    def go_to_position3(self, robot_id):
        GK_defense_x = math.cos(self.defense_angle)*1.2 - self.field[X]/2
        GK_defense_y = math.sin(self.defense_angle)*1.2
        D12_defense_x = 0.12*(self.cur_ball[X] - (self.goal[X] + self.field[X])/2)
        D12_defense_y = 0.12*(self.cur_ball[Y])
        F12_attack_x = 0.12*(self.cur_ball[X] + self.field[X]/6) 
        F12_attack_y = 0.12*(self.cur_ball[Y])
        i = robot_id
        if i != 0:
            idx = helper.find_closest_robot(self.cur_ball,self.cur_posture,5)
            if i == idx:
                D12_defense_x = 0.25*(self.cur_ball[X]+self.cur_posture[i][X])
                D12_defense_y = 0.25*(self.cur_ball[Y]+self.cur_posture[i][Y])
                F12_attack_x = 0.25*(self.cur_ball[X]+self.cur_posture[i][X])
                F12_attack_y = 0.25*(self.cur_ball[Y]+self.cur_posture[i][Y])
        dp = [
            [GK_defense_x, GK_defense_y],
            [D12_defense_x, D12_defense_y],
            [D12_defense_x, D12_defense_y],
            [F12_attack_x, F12_attack_y],
            [F12_attack_x, F12_attack_y],
        ]
        x, y = dp[robot_id]

        return self.go_to(robot_id, x, y)

    def pass_to_robot(self, robot_id):
        if not self.cur_posture[robot_id][BALL_POSSESSION]:
            return self.stop(robot_id)
        if robot_id == 0:
            t1, t2 = 1, 2
        if robot_id == 1:
            t1, t2 = 2, 3
        if robot_id == 2:
            t1, t2 = 1, 4
        if robot_id == 3:
            t1, t2 = 4, 4
        if robot_id == 4:
            t1, t2 = 3, 3
        if helper.distance(self.cur_posture[robot_id][X], self.cur_posture[robot_id][X], self.cur_posture[t1][Y], self.cur_posture[t1][Y]) < 2:
            x, y = self.cur_posture[t1][:2]
        elif helper.distance(self.cur_posture[robot_id][X], self.cur_posture[robot_id][X], self.cur_posture[t2][Y], self.cur_posture[t2][Y]) < 2:
            x, y = self.cur_posture[t2][:2]
        else:
            x, y = self.field[X]/2, 0

        return self.pass_to(robot_id, x, y)

    def cross_to_robot(self, robot_id):
        if not self.cur_posture[robot_id][BALL_POSSESSION]:
            return self.stop(robot_id)
        if robot_id == 0:
            t1, t2 = 1, 2
        if robot_id == 1:
            t1, t2 = 2, 3
        if robot_id == 2:
            t1, t2 = 1, 4
        if robot_id == 3:
            t1, t2 = 4, 4
        if robot_id == 4:
            t1, t2 = 3, 3
        if helper.distance(self.cur_posture[robot_id][X], self.cur_posture[robot_id][X], self.cur_posture[t1][Y], self.cur_posture[t1][Y]) < 2:
            x, y = self.cur_posture[t1][:2]
        elif helper.distance(self.cur_posture[robot_id][X], self.cur_posture[robot_id][X], self.cur_posture[t2][Y], self.cur_posture[t2][Y]) < 2:
            x, y = self.cur_posture[t2][:2]
        else:
            x, y = self.field[X]/2, 0

        wheel = self.cross_to(robot_id, x, y, 0.4)
        if wheel == None:
            wheel = self.pass_to(robot_id, x, y)
        return wheel

    def shoot(self, robot_id):
        if not self.cur_posture[robot_id][BALL_POSSESSION]:
            return self.stop(robot_id)
        x, y = self.field[X]/2, 0
        return self.shoot_to(robot_id, x, y)

    def gkgk(self):
        gk_index = 0
        gk_control = self.defend_ball(gk_index)
        if gk_control == None:
            if -self.field[X]/2 - 0.05 < self.cur_posture[gk_index][X] < -self.field[X]/2 + 0.15 and -0.02 < self.cur_posture[gk_index][Y] < 0.02:
                gk_control = self.turn_to(gk_index, 0, 0)
            else:
                gk_control = self.go_to(gk_index, -self.field[X]/2, 0)
        return gk_control