#!/usr/bin/python3

# Author(s): Luiz Felipe Vecchietti, Kyujin Choi, Taeyoung Kim
# Maintainer: Kyujin Choi (nav3549@kaist.ac.kr)
import random
from collections import namedtuple


Transition = namedtuple('Transition', ('observation','state','init_hidden', 'action', 'reward', 'team_reward', 'mask', 'sammica'))


class Memory(object):
    def __init__(self, capacity):
        self.memory = []
        self.capacity = capacity
        self.position = 0

    def push(self, observation, state, init_hidden, action, reward, team_reward, mask, sammica):

        if len(self.memory) < self.capacity:
            self.memory.append(Transition(observation, state, init_hidden, action, reward, team_reward, mask, sammica))
        self.memory[self.position] = Transition(observation, state, init_hidden, action, reward, team_reward, mask, sammica)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))
        return batch

    def __len__(self):
        return len(self.memory)