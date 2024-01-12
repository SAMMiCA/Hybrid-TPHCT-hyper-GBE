#!/usr/bin/python3
# Maintainer: Kyujin Choi (nav3549@kaist.ac.kr)
from networks import RNNHyperAgent
from qmix import QMixer
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from copy import deepcopy
import numpy as np

import helper
import os
from episode_memory import Memory
import random

CHECKPOINT_GK= os.path.join(os.path.dirname(__file__), 'models/Robot_GK.th')
CHECKPOINT_D12= os.path.join(os.path.dirname(__file__), 'models/Robot_D12.th')
CHECKPOINT_F12= os.path.join(os.path.dirname(__file__), 'models/Robot_F12.th')
CHECKPOINT= [CHECKPOINT_GK, CHECKPOINT_D12, CHECKPOINT_F12]
CHECKPOINT_TEAM_MIXER =  os.path.join(os.path.dirname(__file__), 'models/team_mixer.th')
GK_INDEX = 0
D1_INDEX = 1
D2_INDEX = 2
F1_INDEX = 3
F2_INDEX = 4

def update_target_model(net, target_net):
    target_net.load_state_dict(net.state_dict())

class DRQN:
    def __init__(self, n_agents, dim_obs, dim_mix_state, dim_act, epi_max_len, epsilon=0.95, load=False):
        self.role_type = 3
        self.agents_GK = 1
        self.agents_D12 = 2
        self.agents_F12 = 2
        self.n_agents =  [self.agents_GK, self.agents_D12, self.agents_F12] # this option for multi-agent RL
        self.training_stage = 4

        self.n_states = dim_obs
        self.n_actions = dim_act
        self.epi_max_len = epi_max_len
        self._iterations = 0
        self.discount_factor = 0.99
        self.update_steps = 16000 
        self.team_update_steps = 16000  
        self.epsilon_steps = 10000
        self.epsilon = epsilon # Initial epsilon value         
        self.final_epsilon = 0.05 # Final epsilon value
        self.dec_epsilon = 0.025 # Decrease rate of epsilon for every generation

        self.observation_steps = 300 # Number of iterations to observe before training every generation
        self.batch_size = 64
        self.save_num = 100 # Save checkpoint #defualt is 100

        self.num_inputs = dim_obs
        self.act_size = dim_act
        self.mixer_state = dim_mix_state
        self.mixing_dim = 32
        self.memory = Memory(5000)
        self.gamma = 0.99
        self.grad_norm_clip = 10
        self.hidden_states_GK = None
        self.hidden_states_D12 = None
        self.hidden_states_F12 = None
        self.hidden_states=[self.hidden_states_GK, self.hidden_states_D12, self.hidden_states_F12]
        self.target_hidden_states_GK = None
        self.target_hidden_states_D12 = None
        self.target_hidden_states_F12 = None
        self.target_hidden_states=[self.target_hidden_states_GK, self.target_hidden_states_D12, self.target_hidden_states_F12]
        self.loss = [0 for _ in range(self.training_stage)]

        self.num_states = 2

        self.net = [RNNHyperAgent(self.num_inputs, self.num_states, self.act_size), RNNHyperAgent(self.num_inputs, self.num_states, self.act_size), RNNHyperAgent(self.num_inputs, self.num_states, self.act_size)]
        self.team_mixer = QMixer(self.role_type, self.mixer_state, self.mixing_dim)
        self.load = load
        if self.load == True:
            for role in range(self.role_type):
                self.net[role].load_state_dict(torch.load(CHECKPOINT[role]))
            self.team_mixer.load_state_dict(torch.load(CHECKPOINT_TEAM_MIXER, map_location=lambda storage, loc: storage))
            helper.printConsole("loading variables...")

        self.target_net = copy.deepcopy(self.net)
        self.target_team_mixer = copy.deepcopy(self.team_mixer)



        self.params_GK = list(self.net[0].parameters())
        self.params_D12 = list(self.net[1].parameters())
        self.params_F12 = list(self.net[2].parameters())
        self.team_params = self.params_GK + self.params_D12 + self.params_F12 + list(self.team_mixer.parameters())
        self.params = [self.params_GK, self.params_D12, self.params_F12, self.team_params]



        self.lr = 0.00004
        self.team_lr = 0.0002
        self.optimizer = [optim.Adam(params=self.params[role], lr=self.lr) for role in range(self.training_stage)] 
        self.training_phase = [0,1,2,3]
        self.training_phase_change = 100000 




        for role in range(self.role_type):
            self.net[role].train()
            self.target_net[role].train() 
            self.net[role].cuda()
            self.target_net[role].cuda()
        self.team_mixer.train()
        self.team_mixer.cuda()
        self.target_team_mixer.train()
        self.target_team_mixer.cuda()


    def select_action(self, act_input, sammica):

        device = torch.device("cuda")
        act_input = torch.Tensor(act_input).to(device)
        act_input_GK = act_input[0:1]
        act_input_D12 = act_input[1:3]
        act_input_F12 = act_input[3:5]
        act_input = [act_input_GK, act_input_D12, act_input_F12]

        sammica = np.array([sammica]*sum(self.n_agents))
        sammica = torch.Tensor(sammica).to(device)
        sammica = [sammica[0:1],sammica[1:3],sammica[3:5]]

        out_put_actions = []
        for role in range(self.role_type):
            self.qvalue, self.hidden_states[role] = self.net[role](act_input[role], sammica[role], self.hidden_states[role])
            self.qvalue = self.qvalue.cpu().data.numpy()
            pick_random = int(np.random.rand() <= self.epsilon)
            random_actions = np.random.randint(0, self.act_size, self.n_agents[role])
            picked_actions = pick_random * random_actions + (1 - pick_random) * np.argmax(self.qvalue, axis =1)
            out_put_actions += list(picked_actions)

        return out_put_actions

    def init_hidden(self):
        self.hidden_states = [self.net[role].init_hidden().unsqueeze(0).expand(1, self.n_agents[role], -1) for role in range(self.role_type)]
        self.target_hidden_states = [self.target_net[role].init_hidden().unsqueeze(0).expand(1, self.n_agents[role], -1) for role in range(self.role_type)]

        
   
    def update_policy(self):

        batch = self.memory.sample(self.batch_size)

        device = torch.device("cuda")
        states = torch.Tensor(batch.state).to(device)

        sammicas = torch.Tensor(batch.sammica).to(device)
        sammicas = sammicas.unsqueeze(2).repeat([1,1,5,1])
        sammicas_GK = sammicas[:,:,0:1] 
        sammicas_D12 = sammicas[:,:,1:3] 
        sammicas_F12 = sammicas[:,:,3:5]
        sammicas = [sammicas_GK, sammicas_D12, sammicas_F12] 

        
        mask = torch.Tensor(batch.mask).to(device)
        mask = mask[:, :-1]
        
        rewards = torch.Tensor(batch.reward).to(device)
        team_rewards = torch.Tensor(batch.team_reward).to(device)
        rewards = rewards[:, :-1]
        team_rewards = team_rewards[:, :-1]
        rewards_GK = rewards[:,:,0]
        rewards_D12 = rewards[:,:,1]
        rewards_F12 = rewards[:,:,2]
        rewards = [rewards_GK, rewards_D12, rewards_F12, team_rewards]


        observations = torch.Tensor(batch.observation).to(device)
        observations_GK = observations[:,:,0:1] 
        observations_D12 = observations[:,:,1:3] 
        observations_F12 = observations[:,:,3:5]
        observations = [observations_GK, observations_D12, observations_F12] 

        actions = torch.Tensor(batch.action).long().to(device)
        actions = actions[:, :-1]
        actions_GK = actions[:,:,0:1]
        actions_D12 = actions[:,:,1:3]
        actions_F12 = actions[:,:,3:5]
        actions = [actions_GK, actions_D12, actions_F12]


        init_hidden_states = torch.Tensor(batch.init_hidden).to(device)
        init_hidden_states_GK = init_hidden_states[:,0:1]
        init_hidden_states_D12 = init_hidden_states[:,1:3]
        init_hidden_states_F12 = init_hidden_states[:,3:5]
        init_hidden_states = [init_hidden_states_GK,init_hidden_states_D12,init_hidden_states_F12]



        input_chosen_action_qvals = []
        input_target_max_qvals = []


        for role in range(len(self.training_phase)):
            if role != 3:

                q_out = []
                hidden_states = init_hidden_states[role]
                for i in range(self.epi_max_len):
                    act_input = observations[role][:,i,:,:]
                    sammica = sammicas[role][:,i,:,:]
          
                    act_input= act_input.reshape(self.batch_size*self.n_agents[role], -1)
                    qvalue, hidden_states = self.net[role](act_input, sammica, hidden_states)
                    q_out.append(qvalue.view(self.batch_size,self.n_agents[role],-1))

                q_out = torch.stack(q_out, dim=1)  # Concat over time

          

                # Pick the Q-Values for the actions taken by each agent
                chosen_action_qvals = torch.gather(q_out[:, :-1], dim=3, index=actions[role]).squeeze(3)  # Remove the last dim

                target_q_out = []
                target_hidden_states =  init_hidden_states[role]
                for i in range(self.epi_max_len):
                    act_input = observations[role][:,i,:,:]
                    sammica = sammicas[role][:,i,:,:]
                    act_input= act_input.reshape(self.batch_size*self.n_agents[role], -1)
                    target_qvalue, target_hidden_states = self.target_net[role](act_input, sammica, target_hidden_states)
                    target_q_out.append(target_qvalue.view(self.batch_size,self.n_agents[role],-1))
                    
                # We don't need the first timesteps Q-Value estimate for calculating targets
                target_q_out = torch.stack(target_q_out[1:], dim=1)

                cur_max_actions = q_out[:, 1:].max(dim=3, keepdim=True)[1]
                target_max_qvals = torch.gather(target_q_out, 3, cur_max_actions).squeeze(3)



            
            # Making Q_D12 and Q_F12 when agent is not GK
            if (role == 1) or (role == 2) :
                chosen_action_qvals = torch.sum(chosen_action_qvals, dim=2, keepdim=True)
                target_max_qvals = torch.sum(target_max_qvals, dim=2, keepdim=True)

            if role == 3:
                # Combine Q_GK, Q_D12_tatal, Q_F12_tatal and make Q_Team !!!
                input_chosen_action_qvals = torch.cat((input_chosen_action_qvals[0],input_chosen_action_qvals[1],input_chosen_action_qvals[2]), 2) 
                input_target_max_qvals = torch.cat((input_target_max_qvals[0],input_target_max_qvals[1],input_target_max_qvals[2]), 2) 
                chosen_action_qvals = self.team_mixer(input_chosen_action_qvals, states[:, :-1])
                target_max_qvals = self.target_team_mixer(input_target_max_qvals, states[:, 1:])


            else:
                # Combine Q_GK, Q_D12_tatal, Q_F12_tatal
                input_chosen_action_qvals.append(chosen_action_qvals)
                input_target_max_qvals.append(target_max_qvals)

            # Calculate 1-step Q-Learning targets
            if role == 3:
                targets = (rewards[0] + rewards[1] + rewards[2] + rewards[3]) + self.gamma * target_max_qvals
            else:
                targets = (rewards[role] + 0.33*rewards[3]) + self.gamma * target_max_qvals
            if role in self.training_phase:
                # Td-error
                td_error = (chosen_action_qvals - targets.detach())
                mask = mask.expand_as(td_error)

                # 0-out the targets that came from padded data
                masked_td_error = td_error * mask


                # Normal L2 loss, take mean over actual data
                loss = (masked_td_error ** 2).sum() / mask.sum()
                self.loss[role] = loss.cpu().data.numpy()
              

                self.optimizer[role].zero_grad()
                loss.backward(retain_graph = True)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.params[role], self.grad_norm_clip)
        self.optimizer[0].step()
        self.optimizer[1].step()
        self.optimizer[2].step()
        self.optimizer[3].step()
        
        if self._iterations  % self.update_steps == 0: 
            for role in range(self.role_type):
                update_target_model(self.net[role], self.target_net[role])
            helper.printConsole("Updated target model.")
        if self._iterations  % self.team_update_steps == 0:         
            update_target_model(self.team_mixer, self.target_team_mixer)
            helper.printConsole("Updated team target model.")

        # if self._iterations  % self.epsilon_steps == 0:
        #     self.epsilon = max(self.epsilon - self.dec_epsilon, self.final_epsilon)
        #     helper.printConsole("New Episode! New Epsilon:" + str(self.epsilon))

        return self.loss  



    def print_loss(self, loss, episode):
        if self._iterations % 100 == 0: # Print information every 100 0iterations
            helper.printConsole("======================================================")
            helper.printConsole("Episode: " + str(episode))
            helper.printConsole("Epsilon: " + str(self.epsilon))
            helper.printConsole("Iteration: " + str(self._iterations))
            helper.printConsole("GK_Loss: " + str(loss[0]))
            helper.printConsole("D12_Loss: " + str(loss[1]))
            helper.printConsole("F12_Loss: " + str(loss[2]))
            helper.printConsole("Team_Loss: " + str(loss[3]))
            helper.printConsole("======================================================")

    def update(self, episode):
        if len(self.memory) > self.observation_steps:
            self._iterations += 1
            if episode < self.training_phase_change:
                self.training_phase = [0,1,2,3]
            elif episode >= self.training_phase_change:
                self.training_phase = [0,1,2,3]
                # self.training_phase = [3,10,11,12] # when using Phase transfer
                # self.loss[0]=0 # when using Phase transfer
                # self.loss[1]=0 # when using Phase transfer
                # self.loss[2]=0 # when using Phase transfer
            loss = self.update_policy()
            self.print_loss(loss, episode)


  
        