#!/usr/bin/python3

# Author(s): Luiz Felipe Vecchietti, Kyujin Choi, Taeyoung Kim
# Maintainer: Kyujin Choi (nav3549@kaist.ac.kr)

import random
import os
import sys

import math
import numpy as np

import matplotlib.pyplot as plt

import time
import pickle

data_path = os.path.join(os.path.dirname(__file__), 'MULTI-AGENT-REWARD_data.pickle')
fig_path = os.path.join(os.path.dirname(__file__), 'reward.png')

class Plot():
    def __init__(self):
        self.episode = []
        self.m_episode = []
        self.num_value = 5
        self.value = [[] for _ in range(self.num_value)]
        self.mean_value =  [[] for _ in range(self.num_value)]
        self.init_value = [0,0,0,0,0]
        # self.init_value = [0.45,1.9,1.7,1,5]
        # self.normalize = [1,2,2,1,5]
        self.normalize = [1,1,1,1,1]

        with open(data_path,"rb") as fr:
            self.value = pickle.load(fr)
        self.num = 10


        print("Initializing variables...")

    def run(self):
        # for i in range(10000):
        for i in range(len(self.value[0])):
            self.episode.append(100*(i+1))
            if i >= self.num:
                if self.num == i:
                    for role in range(self.num_value):
                        self.init_value[role] = np.mean(self.value[role][i-self.num:i])/self.normalize[role]
                self.m_episode.append(self.episode[i] - self.num/2)
                for role in range(self.num_value):
                    self.mean_value[role].append( -self.init_value[role] + np.mean(self.value[role][i-self.num:i])/self.normalize[role])
        plt.title('Proposed Method Reward')
        xlab = 'Training Episodes'
        ylab = 'Incresed Reward from Initial Value'
        color=['b','g','c','y','r']
        label=['Average_GK_Reward','Average_D12_Reward','Average_F12_Reward','Average_Team_Reward','Average_Total_Reward']
        for role in range(self.num_value):
            # plt.plot(self.episode, self.value, c = 'lightskyblue', label='total_reward') 
            plt.xticks([ 20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000, 180000, 200000, 220000, 240000, 260000, 280000, 300000, 320000, 340000, 360000, 380000, 400000],
             ['20k', '40k', '60k', '80k', '100k', '120k', '140k', '160k', '180k', '200k', '220k', '240k', '260k', '280k', '300k', '320k', '340k', '360k', '380k', '400k'])
            plt.ylim( -0.1, 4.0)
            plt.xlabel(xlab)
            plt.ylabel(ylab)
            plt.plot(self.m_episode, self.mean_value[role], c = color[role], label= label[role]) 
        plt.legend(loc=2)
        plt.grid(True)
        fig_path = os.path.join(os.path.dirname(__file__), 'Proposed_Method_Reward.png')
        plt.savefig(fig_path)
        print("plot")
        

if __name__ == '__main__':
    main = Plot()
    main.run()