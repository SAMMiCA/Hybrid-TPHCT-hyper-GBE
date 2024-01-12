#!/usr/bin/python3

# Author(s): Luiz Felipe Vecchietti, Kyujin Choi, Taeyoung Kim
# Maintainer: Kyujin Choi (nav3549@kaist.ac.kr)

import random
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../common')
try:
    from participant import Participant, Game, Frame
except ImportError as err:
    print('player_random-walk: \'participant\' module cannot be imported:', err)
    raise

import pickle
import torch

import math
import numpy as np

import helper
from action import ActionControl
from drqn import DRQN
from rl_utils import  Logger, get_hybrid_action, get_reward, get_team_reward, get_state, get_global_state

from perception.mlp_module import PerceptionModule
from reasoning.crn_module import ReasoningModule


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

#rodot_index
GK_INDEX = 0 
D1_INDEX = 1 
D2_INDEX = 2 
F1_INDEX = 3 
F2_INDEX = 4



config_path = os.path.join(os.path.dirname(__file__), 'config.pickle')
memory_path = os.path.join(os.path.dirname(__file__), 'memory.pickle')

class Frame(object):
    def __init__(self):
        self.time = None
        self.score = None
        self.reset_reason = None
        self.game_state = None
        self.subimages = None
        self.coordinates = None
        self.half_passed = None

class TestPlayer(Participant):
    def init(self, info):
        self.field = info['field']
        self.max_linear_velocity = info['max_linear_velocity']
        self.goal = info['goal']
        self.penalty_area = info['penalty_area']
        self.goal_area = info['goal_area']
        self.number_of_robots = info['number_of_robots']
        self.end_of_frame = False
        self._frame = 0 
        self.wheels = [ 0 for _ in range(30)]
        self.cur_posture = []
        self.prev_posture = []
        self.cur_posture_opp = []
        self.cur_ball = []
        self.prev_ball = []

        self.previous_frame = Frame()
        self.frame_skip = 2 # number of frames to skip
        self.epi_max_len = 40
        self.obs_size = 32 #243 #37 for usual state #243 for lidar state
        self.state_size = 22
        self.act_size = 6
        self.role_type = 3
        self.mixer_num = 2
        self.episode_observation = [[[0 for _ in range(self.obs_size)] for _ in range(self.number_of_robots)] for _ in range(self.epi_max_len)]
        self.episode_state = [[0 for _ in range(self.state_size)] for _ in range(self.epi_max_len)]
        self.episode_sammica = [[0 for _ in range(2)] for _ in range(self.epi_max_len)]
        self.episode_action = [[0 for _ in range(self.number_of_robots)] for _ in range(self.epi_max_len)]
        self.episode_reward = [[0 for _ in range(self.role_type)] for _ in range(self.epi_max_len)]
        self.episode_team_reward = [0 for _ in range(self.epi_max_len)]
        self.episode_mask = [0 for _ in range(self.epi_max_len)]
        self.t = 0
        # for RL
        self.ActionControl = ActionControl(self.max_linear_velocity, self.field, self.goal)
        self.action = [0 for _ in range(self.number_of_robots)]
        self.pre_action = [0 for _ in range(self.number_of_robots)]

        self.num_inputs = self.obs_size
        self.load = False
        self.epsilon = 0
        self.op_episode = 0
        self.episode = 0

        self.total_reward = 0
        self.reward = [0 for _ in range(self.number_of_robots)]
        self.team_reward = 0
        self.rew =np.zeros((self.number_of_robots,4))

        self.trainer = DRQN(self.number_of_robots, self.obs_size, self.state_size, self.act_size, self.epi_max_len, self.epsilon, self.load)
        self.trainer.init_hidden()

        env = ['friction', '4vs5', 'malfunction']
        self.env_num = 0
        CHECKPOINT_GK = os.path.join(os.path.dirname(__file__), 'models/'+env[self.env_num]+'/Robot_GK.th')
        CHECKPOINT_D12 = os.path.join(os.path.dirname(__file__), 'models/'+env[self.env_num]+'/Robot_D12.th')
        CHECKPOINT_F12 = os.path.join(os.path.dirname(__file__), 'models/'+env[self.env_num]+'/Robot_F12.th')
        CHECKPOINT = [CHECKPOINT_GK, CHECKPOINT_D12, CHECKPOINT_F12]
        for role in range(self.role_type):
            self.trainer.net[role].load_state_dict(torch.load(CHECKPOINT[role]))

        self.trainer2 = DRQN(self.number_of_robots, self.obs_size, self.state_size, self.act_size, self.epi_max_len, self.epsilon, self.load)
        self.trainer2.init_hidden()
        CHECKPOINT_GK = os.path.join(os.path.dirname(__file__), 'models/pretrain/'+'Robot_GK.th')
        CHECKPOINT_D12 = os.path.join(os.path.dirname(__file__), 'models/pretrain/'+'Robot_D12.th')
        CHECKPOINT_F12 = os.path.join(os.path.dirname(__file__), 'models/pretrain/'+'Robot_F12.th')
        CHECKPOINT2 = [CHECKPOINT_GK, CHECKPOINT_D12, CHECKPOINT_F12]
        for role in range(self.role_type):
            self.trainer2.net[role].load_state_dict(torch.load(CHECKPOINT2[role]))
        
        # Perception Module
        self.perception_module = PerceptionModule()
        # Reasoning Module
        self.reasoning_module = ReasoningModule()

        self.printConsole("Initializing variables...")

    def get_coord(self, received_frame):
        self.cur_ball = received_frame.coordinates[BALL]
        self.cur_posture = received_frame.coordinates[MY_TEAM]
        self.cur_posture_opp = received_frame.coordinates[OP_TEAM]
        self.prev_posture = self.previous_frame.coordinates[MY_TEAM]
        self.prev_posture_opp = self.previous_frame.coordinates[OP_TEAM]
        self.prev_ball = self.previous_frame.coordinates[BALL]

    def init_episode(self):
        self.episode_observation = [[[0 for _ in range(self.obs_size)] for _ in range(self.number_of_robots)] for _ in range(self.epi_max_len)]
        self.episode_state = [[0 for _ in range(self.state_size)] for _ in range(self.epi_max_len)]
        self.episode_sammica = [[0 for _ in range(2)] for _ in range(self.epi_max_len)]
        self.episode_action = [[0 for _ in range(self.number_of_robots)] for _ in range(self.epi_max_len)]
        self.episode_reward = [[0 for _ in range(self.role_type)] for _ in range(self.epi_max_len)]
        self.episode_team_reward = [0 for _ in range(self.epi_max_len)]
        self.episode_mask = [0 for _ in range(self.epi_max_len)]
        self.t = 0

    def sammica(self, received_frame):
        if self.env_num < 2:
            # Perception Module
            # in: received frame and speeds
            # out: friction label and latent vector to be used for reasoning module
            perception, latent = self.perception_module.perceive(received_frame, self.wheels)
            # Result 0: Friction 3.0
            # Result 1: Friction 0.1
            # Result 2: Friction 0.5
            if perception is None:  # perception module returns None when not enough frames have been seen
                friction = None
            else:
                friction = [3.0, 0.1, 0.5][perception]
            # self.printConsole(
            #     "Perception Label: {}, which is Friction {}".format(perception, friction))

            # Reasoning Module
            # in: latent vector from perception module (should be converted to numpy array)
            # out: cluster label and corresponding centroid vector
            if latent is None:
                reason, centroid = None, None
            else:
                reason, centroid = self.reasoning_module.reason(latent.cpu().numpy())
            # self.printConsole(
            #     "Reasoning Label: {}".format(reason))

            if perception == None:
                perception = 3
            if reason == None:
                reason = 14
            result = np.array([round((perception - 1.5)/1.5,1), round((reason-7)/7,1)])

        else: # malfunction
            if len(received_frame.breakdown) == 0:
                bd_id, bd_type = 0, 0
            else:
                bd_id, bd_type = received_frame.breakdown[0]

                bd_id = bd_id + 1
                bd_type = bd_type + 1

            bd_id = round((bd_id - 2.5)/2.5,1)
            bd_type = round((bd_type + 1 - 2)/2,1)

            result = np.array([bd_id, bd_type])

        return result

    def update(self, received_frame):

        if received_frame.end_of_frame:
        
            self._frame += 1

            if (self._frame == 1):
                self.previous_frame = received_frame
                self.get_coord(received_frame)

            self.get_coord(received_frame)
            self.ActionControl.update_state(self.cur_posture, self.prev_posture, self.cur_ball, self.prev_ball, received_frame.reset_reason)

            ## episode ##
            if self._frame % self.frame_skip == 1:
                state = get_state(self.cur_posture, self.prev_posture, self.cur_posture_opp, self.prev_posture_opp, self.cur_ball, self.prev_ball, self.field, self.goal, self.max_linear_velocity) # when use state
                global_state = get_global_state(self.cur_posture, self.prev_posture, self.cur_posture_opp, self.prev_posture_opp, self.cur_ball, self.prev_ball, self.field, self.goal, self.max_linear_velocity) # when use state
                sammica_result = self.sammica(received_frame)
                
                for i in range(self.number_of_robots):
                    self.episode_observation[self.t][i] = state[i] # when use state

                self.episode_observation = np.reshape([self.episode_observation],(self.epi_max_len, self.number_of_robots, self.obs_size))
                
                self.episode_state[self.t] = global_state
                self.episode_state = np.reshape([self.episode_state],(self.epi_max_len, self.state_size))

                self.episode_sammica[self.t] = sammica_result
                self.episode_sammica = np.reshape([self.episode_sammica],(self.epi_max_len, 2))

                act_input = self.episode_observation[self.t]
                self.action = self.trainer.select_action(act_input, self.episode_sammica[self.t])

                action2 = self.trainer2.select_action(act_input, np.array([-1.0,-1.0]))

                if self.env_num == 0:
                    s = 0
                    for i in range(self.number_of_robots):
                        s += self.cur_posture[0][ACTIVE]
                    if s == 5:
                        self.action = action2
                if self.env_num == 1:
                    if sammica_result[0] == -1:
                        self.action = action2
                if self.env_num == 2:
                    id = self.episode_sammica[self.t][0]*2.5+2.5 - 1
                    if id == 0:
                        self.action[1] = action2[1]
                        self.action[2] = action2[2]
                        self.action[3] = action2[3]
                        self.action[4] = action2[4]
                    elif id in [1, 2]:
                        self.action[0] = action2[0]
                        self.action[3] = action2[3]
                        self.action[4] = action2[4]
                    elif id in [3, 4]:
                        self.action[0] = action2[0]
                        self.action[1] = action2[1]
                        self.action[2] = action2[2]

            else:
                self.action = self.pre_action

            ## set speed ##
            for id in range(self.number_of_robots):
                self.wheels[6*id:6*id+6] = get_hybrid_action(self.ActionControl, id, self.action[id], self.cur_posture[id][BALL_POSSESSION])
            self.set_speeds(self.wheels)

            ## episode end ##
            if (received_frame.reset_reason > 1):
                self.trainer.init_hidden()
                self.trainer2.init_hidden()

            self.end_of_frame = False
            self.pre_action = self.action
            self.previous_frame = received_frame

if __name__ == '__main__':
    player = TestPlayer()
    player.run()