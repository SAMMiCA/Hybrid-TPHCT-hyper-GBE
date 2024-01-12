#!/usr/bin/python3

# Author(s): Luiz Felipe Vecchietti, Kyujin Choi, Taeyoung Kim
# Maintainer: Kyujin Choi (nav3549@kaist.ac.kr)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys

class RNNAgent(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(RNNAgent, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.rnn_hidden_dim = 128

        # self.fc0 =  TimeDistributed(self.num_inputs)
        # print(self.fc0)
        self.fc1 = nn.Linear(self.num_inputs, self.rnn_hidden_dim)
        self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.fc2 = nn.Linear(self.rnn_hidden_dim, self.num_outputs)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.rnn_hidden_dim).zero_()
    
    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h

    def save_model(self, net, filename):
        torch.save(net.state_dict(), filename)

class RNNHyperAgent(nn.Module):
    def __init__(self, num_inputs, num_states, num_outputs):
        super(RNNHyperAgent, self).__init__()
        self.num_inputs = num_inputs
        self.num_states = num_states
        self.num_outputs = num_outputs
        self.rnn_hidden_dim = 128

        self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)

        # self.embed_dim = 32
        self.embed_dim = self.rnn_hidden_dim
        self.hyper_w_1 = nn.Linear(self.num_states, self.embed_dim * self.num_inputs)
        self.hyper_w_final = nn.Linear(self.num_states, self.embed_dim * self.num_outputs)

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.num_states, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.num_states, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, self.num_outputs))


    def init_hidden(self):
        # make hidden states on same device as model
        return torch.tensor(np.zeros((1,self.rnn_hidden_dim))).type(torch.FloatTensor)

    def forward(self, inputs, states, hidden_state):
        bs = inputs.size(0)
        states = states.reshape(-1, self.num_states)
        inputs = inputs.view(-1, 1, self.num_inputs)
        # First layer
        w1 = torch.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.num_inputs, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(inputs, w1) + b1)

        # rnn
        hidden = hidden.reshape(-1, self.rnn_hidden_dim)
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim).to("cuda")
        h = self.rnn(hidden, h_in)
        h = h.reshape(-1, 1, self.rnn_hidden_dim)

        # Second layer
        w_final = torch.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, self.num_outputs)
        # State-dependent bias
        v = self.V(states).view(-1, 1, self.num_outputs)
        # Compute final output
        y = torch.bmm(h, w_final) + v
        # Reshape and return
        q = y.view(bs, self.num_outputs)
        return q, h

    def save_model(self, net, filename):
        torch.save(net.state_dict(), filename)

