# AI for Autonomous Vehicles - Build a Self-Driving Car

# Importing the libraries

import os 
import random
import torch
import torch.nn as nn 
import torch.nn.functional as F  
import torch.optim as optim 
from torch.autograd import Variable  

# Creating the architecture of the Neural Network

#input_size = no of input neuron,
#nb_action is no of output neurons
class Network(nn.Module):
    def __init__(self, input_size, nb_action):
        #super used to get the functions into the self instance of the class
        super(Network, self).__init__()
        # attact to the object variables the parameters.
        self.input_size = input_size
        self.nb_action = nb_action
        # finally we use the linear functions to make the connections
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, nb_action)
     #self to us the variables , inputs states are the actuall states where input_size was the number of states.
    def forward(self, state):
        # F is the functional module of the neural network, relu is retifier
        # we input the sate into the first full connection 
        # we use self as its part of the instance of the object 
        x = F.relu(self.fc1(state))
        # finall we output neurons which is the Q_values 
        q_values = self.fc2(x)        
        #finally we o/p q_values 
        return q_values

# Implementing Experience Replay

class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    #memory should not exceed 1000 or more as it will go into not responding mode, and if it happens, we have to force close the app
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    #the zip function will be taking care of  what torch.cat does -> it turns this normal array intp a torch varriable, so we wrap with a Variable class for gradient decent 
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)

# Implementing Deep Q-Learning

class Dqn(object):
    
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(capacity = 100000)
        self.optimizer = optim.Adam(params = self.model.parameters())
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
    
    def select_action(self, state):
        #softmax function helps us explore the enviromnet a little bit using temprature  #it give back probablity 
        #we get the values for the softmax from the model which gives back the q values 
        #we get the Qvalue as when we give the model a Variable packed sate 
        #which convest tensor to gradient decent
        probs = F.softmax(self.model(Variable(state))*100)
        action = probs.multinomial(len(probs))
        return action.data[0,0]
    
    def learn(self, batch_states, batch_actions, batch_rewards, batch_next_states):
        batch_outputs = self.model(batch_states).gather(1, batch_actions.unsqueeze(1)).squeeze(1)
        batch_next_outputs = self.model(batch_next_states).detach().max(1)[0]
        batch_targets = batch_rewards + self.gamma * batch_next_outputs
        td_loss = F.smooth_l1_loss(batch_outputs, batch_targets)
        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()
    
    def update(self, new_state, new_reward):
        new_state = torch.Tensor(new_state).float().unsqueeze(0)
        self.memory.push((self.last_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward]), new_state))
        new_action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_states, batch_actions, batch_rewards, batch_next_states = self.memory.sample(100)
            self.learn(batch_states, batch_actions, batch_rewards, batch_next_states)
        self.last_state = new_state
        self.last_action = new_action
        self.last_reward = new_reward
        return new_action
    def save(self):
        torch.save({"state_dict": self.model.state_dict(), "optimizer": self.optimizer.state_dict()}, "last_brain.pth")
    def load(self):
        if os.path.isfile("last_brain.pth"):
            print("=> loading checkpoint")