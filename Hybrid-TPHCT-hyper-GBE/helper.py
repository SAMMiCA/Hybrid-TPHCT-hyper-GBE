#!/usr/bin/python3

# Author(s): Luiz Felipe Vecchietti, Kyujin Choi, Taeyoung Kim
# Maintainer: Kyujin Choi (nav3549@kaist.ac.kr)

import sys

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
import math

def go_to(id, x, y, cur_posture, cur_ball, max_linear_velocity):
    sign = 1
    kd = 7 if ((id == 1) or (id == 2)) else 5
    ka = 0.4

    tod = 0.005 # tolerance of distance
    tot = math.pi/360 # tolerance of theta

    dx = x - cur_posture[id][X]
    dy = y - cur_posture[id][Y]
    d_e = math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))
    desired_th = math.atan2(dy, dx)

    d_th = wrap_to_pi(desired_th - cur_posture[id][TH])
    
    if (d_th > degree2radian(90)):
        d_th -= math.pi
        sign = -1
    elif (d_th < degree2radian(-90)):
        d_th += math.pi
        sign = -1

    if (d_e < tod):
        kd = 0
    if (abs(d_th) < tot):
        ka = 0

    if go_fast(id, cur_posture, cur_ball):
        kd *= 5

    left_wheel, right_wheel = set_wheel_velocity(max_linear_velocity,
                  sign * (kd * d_e - ka * d_th), 
                  sign * (kd * d_e + ka * d_th))

    return left_wheel, right_wheel

def wrap_to_pi(theta):
    while (theta > math.pi):
        theta -= 2 * math.pi
    while (theta < -math.pi):
        theta += 2 * math.pi
    return theta

def distance(x1, x2, y1, y2):
    return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))

def degree2radian(deg):
    return deg * math.pi / 180

def radian2degree(rad):
    return rad * 180 / math.pi

def predict_ball(cur_ball, previous_ball, prediction_step):
    dx = cur_ball[X] - previous_ball[X]
    dy = cur_ball[Y] - previous_ball[Y]
    predicted_ball = [cur_ball[X] + prediction_step*dx, cur_ball[Y] + prediction_step*dy]
    return predicted_ball

def predict_ball_distance11(self, cur_ball, previous_ball, distance):
    prediction_step = 4*(1/(1+math.exp(-10*distance)))-2
    dx = cur_ball[X] - previous_ball[X]
    dy = cur_ball[Y] - previous_ball[Y]
    predicted_ball = [cur_ball[X] + prediction_step*dx, cur_ball[Y] + prediction_step*dy]
    return predicted_ball

def predict_ball_distance(self, cur_ball, previous_ball, distance):
    prediction_step = 4*(1/(1+math.exp(-1.2*distance)))-2
    dx = cur_ball[X] - previous_ball[X]
    dy = cur_ball[Y] - previous_ball[Y]
    predicted_ball = [cur_ball[X] + prediction_step*dx, cur_ball[Y] + prediction_step*dy]
    # if(predicted_ball[X] < -self.field[X]/2):
    #     predicted_ball[X] = 2*(-self.field[X]/2) - predicted_ball[X]
    # elif(predicted_ball[X] >self.field[X]/2):
    #     predicted_ball[X] = 2*(self.field[X]/2) - predicted_ball[X]

    # if(predicted_ball[Y] < -self.field[Y]/2):
    #     predicted_ball[Y] = 2*(-self.field[Y]/2) - predicted_ball[Y]
    # elif(predicted_ball[Y] >self.field[Y]/2):
    #     predicted_ball[Y] = 2*(self.field[Y]/2) - predicted_ball[Y]

    return predicted_ball

def find_closest_robot(cur_ball, cur_posture, number_of_robots):
    min_idx = 0
    min_distance = 9999.99
    for i in range(number_of_robots):
        measured_distance = distance(cur_ball[X], cur_posture[i][X], cur_ball[Y], cur_posture[i][Y])
        if (measured_distance < min_distance):
            min_distance = measured_distance
            min_idx = i
    if (min_idx == 0):
        idx = 1
    else:
        idx = min_idx
    return idx

def find_closest_action(cur_action, discrete_actions, number_of_actions):
    min_idx = 0
    min_distance = 9999.99
    for i in range(number_of_actions):
        measured_distance = distance(cur_action[X], discrete_actions[i][X], cur_action[Y], discrete_actions[i][Y])
        if (measured_distance < min_distance):
            min_distance = measured_distance
            min_idx = i
    return min_idx

def ball_is_own_goal(predicted_ball, field, goal_area):
    return (-field[X]/2 <= predicted_ball[X] <= -field[X]/2 + goal_area[X] and
            -goal_area[Y]/2 <= predicted_ball[Y] <= goal_area[Y]/2)

def ball_is_own_penalty(predicted_ball, field, penalty_area):
    return (-field[X]/2 <= predicted_ball[X] <= -field[X]/2 + penalty_area[X] and
        -penalty_area[Y]/2 <= predicted_ball[Y] <=  penalty_area[Y]/2)

def ball_is_own_field(predicted_ball):
    return (predicted_ball[X] <= 0)

def ball_is_opp_goal(predicted_ball, field, goal_area):
    return (field[X]/2  - goal_area[X] <= predicted_ball[X] <= field[X]/2 and
            -goal_area[Y]/2 <= predicted_ball[Y] <= goal_area[Y]/2)

def ball_is_opp_penalty(predicted_ball, field, penalty_area):
    return (field[X]/2  - penalty_area[X] <= predicted_ball[X] <= field[X]/2 and
            -penalty_area[Y]/2 <= predicted_ball[Y] <= penalty_area[Y]/2)

def ball_is_opp_field(predicted_ball):
    return (predicted_ball[X] > 0)

def get_defense_kick_angle(predicted_ball, field, cur_ball):
    if predicted_ball[X] >= -field[X] / 2:
        x = -field[X] / 2 - predicted_ball[X]
    else:
        x = -field[X] / 2 - cur_ball[X]
    y = predicted_ball[Y]
    return math.atan2(y, abs(x) + 0.00001)

def get_attack_kick_angle(predicted_ball, field):
    x = field[X] / 2 - predicted_ball[X] + 0.00001
    y = predicted_ball[Y]
    angle = math.atan2(y, x)
    return -angle

def set_wheel_velocity(max_linear_velocity, left_wheel, right_wheel):
    ratio_l = 1
    ratio_r = 1

    if (left_wheel > max_linear_velocity or right_wheel > max_linear_velocity):
        diff = max(left_wheel, right_wheel) - max_linear_velocity
        left_wheel -= diff
        right_wheel -= diff
    if (left_wheel < -max_linear_velocity or right_wheel < -max_linear_velocity):
        diff = min(left_wheel, right_wheel) + max_linear_velocity
        left_wheel -= diff
        right_wheel -= diff

    return left_wheel, right_wheel

def is_facing_target(id, x, y, cur_posture, div):
    dx = x - cur_posture[id][X]
    dy = y - cur_posture[id][Y]
    desired_th = math.atan2(dy, dx)
    d_th = wrap_to_pi(desired_th - cur_posture[id][TH])

    if (d_th > degree2radian(90)):
        d_th -= math.pi
    elif (d_th < degree2radian(-90)):
        d_th += math.pi
    if abs(d_th) < math.pi / div:
        return True
    return False

def shoot_chance(self, id, cur_posture, ball):
    d2b = distance(ball[X], cur_posture[id][X],
                                ball[Y],cur_posture[id][Y])
    dx = ball[X] - cur_posture[id][X]
    dy = ball[Y] - cur_posture[id][Y]

    gy = self.goal_area[Y]

    if (dx < 0) or (d2b > self.field[Y]/2):
        return False

    y = (self.field[X]/2 - ball[X])*dy/dx + cur_posture[id][Y]

    if (abs(y) < gy/2):
        return True
    elif (ball[X] < 2.5) and (self.field[Y] - gy/2 < abs(y) < self.field[Y] + gy/2):
        return True
    else:
        return False

def kickaway_chance(self, id, cur_posture, ball):
    dx = ball[X] - cur_posture[id][X]
    dy = ball[Y] - cur_posture[id][Y]

    gy = self.goal_area[Y]

    if (dx > 0):
        return True

    y = (-self.field[X]/2 - ball[X])*dy/dx + cur_posture[id][Y]

    if (abs(y) < 1.5*gy/2):
        return False
    elif (self.field[Y] - 1.5*gy/2 < abs(y) < self.field[Y] + 1.5*gy/2):
        return True
    else:
        return True

def is_stuck(self, id, cur_posture, cur_posture_opp, number_of_robots):
    # cur_posture, prev_posture, count
    # or coming toward, facing
    # or closest robot, distance
    min_idx = 0
    min_idx_opp = 0
    min_distance = 9999.99
    min_distance_opp = 9999.99
    for i in range(number_of_robots):
        if (i == id):
            measured_distance = 9999.99
        else:
            measured_distance = distance(cur_posture[id][X], cur_posture[i][X], cur_posture[id][Y], cur_posture[i][Y])
        if (measured_distance < min_distance):
            min_distance = measured_distance
            min_idx = i
    for j in range(number_of_robots):
        measured_distance_opp = distance(cur_posture[id][X], cur_posture_opp[j][X], cur_posture[id][Y], cur_posture_opp[j][Y])
        if (measured_distance_opp < min_distance_opp):
            min_distance_opp = measured_distance_opp
            min_idx_opp = j

    # stuck with the wall
    if (cur_posture[id][Y] > 2.245):
        if (abs(cur_posture[id][TH] - math.pi/2) < math.pi/180):
                        return 1, True
        elif (abs(cur_posture[id][TH] + math.pi/2) < math.pi/180):
            return 2, True
    elif (cur_posture[id][Y] < -2.245):
        if (abs(cur_posture[id][TH] - math.pi/2) < math.pi/180):
            return 2, True
        elif (abs(cur_posture[id][TH] + math.pi/2) < math.pi/180):
            return 1, True

    if (min_distance > min_distance_opp):
        if (min_distance_opp < 1.2*robot_size and is_facing_target(id, cur_posture_opp[min_idx_opp][X], cur_posture_opp[min_idx_opp][Y], cur_posture, 6)):
            dx = cur_posture_opp[min_idx_opp][X] - cur_posture[id][X]
            dy = cur_posture_opp[min_idx_opp][Y] - cur_posture[id][Y]
            th = (cur_posture[id][TH] if (min_distance_opp == 0) else math.atan2(dy, dx))
            theta = abs(cur_posture[id][TH]-th)
            return theta, True
    else:
        if (min_distance < 1.2*robot_size and is_facing_target(id, cur_posture[min_idx][X], cur_posture[min_idx][Y], cur_posture, 6)):
            dx = cur_posture[min_idx][X] - cur_posture[id][X]
            dy = cur_posture[min_idx][Y] - cur_posture[id][Y]
            th = (cur_posture[id][TH] if (min_distance == 0) else math.atan2(dy, dx))
            theta = abs(cur_posture[id][TH]-th)
            return theta, True
    return 0, False

def is_full_penalty(self, id, cur_posture):
    if ball_is_opp_penalty(cur_posture[id],self.field, self.penalty_area):
        return False
    else:
        num_in_penalty = 0
        for i in range(5):
            if ball_is_opp_penalty(cur_posture[i], self.field, self.penalty_area):
                num_in_penalty += 1
        if (num_in_penalty >= 2):
            # print(id, "full")
            # sys.__stdout__.flush()
            return True
        else:
            return False

def go_fast(self, id, cur_posture, cur_ball):
    distnace2ball = distance(cur_ball[X], cur_posture[id][X],
                                cur_ball[Y], cur_posture[id][Y])
    d_bg = distance(cur_ball[X], 3.9,
                                cur_ball[Y], 0)
    d_rg = distance(3.9, cur_posture[id][X],
                                0, cur_posture[id][Y])
    
    if (distnace2ball < 0.25 and d_rg > d_bg):
        if (cur_ball[X] > 3.7 and abs(cur_ball[Y]) > 0.5 and abs(cur_posture[id][TH]) < 30 * math.pi/180):
            return False
        else:
            return True
    else:
        return False

def get_ellipse_point(a, b, x_s, y_s, angle):
    t = math.tan(angle)
    if (abs(angle) < math.pi/2):
        x = math.sqrt(a*a*b*b/(a*a*t*t+b*b))
    else:
        x = -math.sqrt(a*a*b*b/(a*a*t*t+b*b))
    y = t * x + y_s
    x += x_s
    return x, y

def is_between(x1, y1, x2, y2, x, y, tolerance):
    if ((((x1 <= x2) and (x1 - tolerance < x < x2 + tolerance))
            or((x1 > x2) and (x1 + tolerance > x > x2 - tolerance)))
        and (((y1 <= y2) and (y1 - tolerance < y < y2 + tolerance))
            or((y1 > y2) and (y1 + tolerance > y > y2 - tolerance)))):
        return True
    else:
        return False

def get_receive_point(id, cur_posture, previous_ball, cur_ball):
    dx = cur_ball[X] - previous_ball[X]
    dy = cur_ball[Y] - previous_ball[Y]

    if (dx == 0):
        dx = 0.0001

    th = math.atan2(dy, dx)

    if (cur_posture[id][Y] > 0):
        y = 1.4
    else:
        y = -1.4

    if dy == 0:
        x = cur_posture[id][X]
    else:
        x = (y - cur_ball[Y])*dx/dy + cur_ball[X]

    if x > 3.9:
        x = 7.8 - x

    return x, y, th

def ball_coming_toward_robot(id, cur_posture, previous_ball, cur_ball):
    x_dir = abs(cur_posture[id][X] - previous_ball[X]) > abs(cur_posture[id][X] - cur_ball[X])
    y_dir = abs(cur_posture[id][Y] - previous_ball[Y]) > abs(cur_posture[id][Y] - cur_ball[Y])
    v_b = distance(cur_ball[X], previous_ball[X], cur_ball[Y], previous_ball[Y])

    if (x_dir and y_dir):
        return True, v_b
    else:
        return False, v_b

def printConsole(message):
    print(message)
    sys.__stdout__.flush()