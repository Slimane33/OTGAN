#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 15:14:07 2020

@author: slimane
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from keras.datasets import mnist
from time import time
import ot


class Generator(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, output_shape):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size_1)
        self.map2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.map3 = nn.Linear(hidden_size_2, output_size)
        self.f = nn.LeakyReLU(negative_slope=0.2)
        self.output_shape = output_shape

    def forward(self, x):
        x = self.f(self.map1(x))
        x = self.f(self.map2(x))
        x = torch.sigmoid(self.map3(x))
        return torch.reshape(x, (-1,)+self.output_shape)

class Critic(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
        super(Critic, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size_1)
        self.map2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.map3 = nn.Linear(hidden_size_2, output_size)
        self.f = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = nn.Flatten()(x)
        x = self.f(self.map1(x))
        x = self.f(self.map2(x))
        return self.map3(x)
    

    
class GAN():
    
    def __init__(self, X_train, gen_params, critic_params):
        self.X_train = X_train

        # default parameters for mnist 
        self.img_rows = X_train.shape[1]
        self.img_cols = X_train.shape[2]
        self.img_channels = X_train.shape[3]
        self.img_shape = (self.img_rows, self.img_cols, self.img_channels)
        self.z_dim = 32
        
        self.generator = Generator(*gen_params)
        self.critic = Critic(*disc_params)
        
            
    
    def sample_data(self, n_sample=100):
        z_random = np.random.randn(n_sample, self.z_dim)
        samples = self.generator(torch.FloatTensor(z_random))
        samples = samples.detach().numpy()
        return samples
    
    
    def sinkhorn(self, a, b, C, reg=0.001, max_iters=100):
        
        K = torch.exp(-C/reg)
        u = torch.ones_like(a)
        v = torch.ones_like(b)
        for i in range(max_iters):
            u = a / torch.matmul(K,v)
            v = b / torch.matmul(K.T,u)
        return torch.matmul(torch.diag_embed(u), torch.matmul(K, torch.diag_embed(v)))
    
    def cost(self, batch_1, batch_2):
        norm_1 = torch.norm(batch_1, p=2, dim=1).reshape(-1,1)
        norm_2 = torch.norm(batch_2, p=2, dim=1).reshape(-1,1)
        return - torch.matmul(batch_1, batch_2.transpose(0,1)) / (torch.matmul(norm_1, norm_2.transpose(0,1))) + 1
        
        
    
    
    def train(self, epochs=1000, batch_size=128, print_interval=1):
        
        criterion = nn.BCELoss()  # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss
        c_optimizer = optim.Adam(self.critic.parameters(), lr=0.0001)
        g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0001)
        d_steps = 1
        g_steps = 1
        
        
        for epoch in range(epochs):
            
            t=time()
            loss_to_display = []
            
            for i in range(self.X_train.shape[0]//batch_size):
                
                self.critic.zero_grad()
                self.generator.zero_grad()
                
                idx_1 = np.random.choice(self.X_train.shape[0], size=batch_size, replace=False)
                idx_2 = np.random.choice(self.X_train.shape[0], size=batch_size, replace=False)
                
                real_1 = torch.FloatTensor(X_train[idx_1])
                real_2 = torch.FloatTensor(X_train[idx_2])
                
                fake_1 = self.generator(torch.FloatTensor(np.random.randn(batch_size, self.z_dim)))
                fake_2 = self.generator(torch.FloatTensor(np.random.randn(batch_size, self.z_dim)))
                
                critic_real_1 = self.critic(real_1)
                critic_real_2 = self.critic(real_2)
                critic_fake_1 = self.critic(fake_1)
                critic_fake_2 = self.critic(fake_2)
                
                # Computing all matrices of costs
                
                costs = torch.zeros((4, 4, batch_size, batch_size))
                
                costs[0,1] = self.cost(critic_real_1, critic_real_2)
                costs[0,2] = self.cost(critic_real_1, critic_fake_1)
                costs[0,3] = self.cost(critic_real_1, critic_fake_2)
                costs[1,2] = self.cost(critic_real_2, critic_fake_1)
                costs[1,3] = self.cost(critic_real_2, critic_fake_2)
                costs[2,3] = self.cost(critic_fake_1, critic_fake_2)
                
                # Computing optimal plans for all costs
                
                a = torch.ones(batch_size) / batch_size
                b = torch.ones(batch_size) / batch_size
                
                plans = torch.zeros((4,4, batch_size, batch_size))
                
                plans[0,1] = self.sinkhorn(a, b, costs[0,1], reg=0.01)
                plans[0,2] = self.sinkhorn(a, b, costs[0,2], reg=0.01)
                plans[0,3] = self.sinkhorn(a, b, costs[0,3], reg=0.01)
                plans[1,2] = self.sinkhorn(a, b, costs[1,2], reg=0.01)
                plans[1,3] = self.sinkhorn(a, b, costs[1,3], reg=0.01)
                plans[2,3] = self.sinkhorn(a, b, costs[2,3], reg=0.01)
                
                # Computing losses
                
                losses = torch.zeros((4,4))
                
                losses[0,1] = torch.sum(plans[0,1] * costs[0,1])
                losses[0,2] = torch.sum(plans[0,2] * costs[0,2])
                losses[0,3] = torch.sum(plans[0,3] * costs[0,3])
                losses[1,2] = torch.sum(plans[1,2] * costs[1,2])
                losses[1,3] = torch.sum(plans[1,3] * costs[1,3])
                losses[2,3] = torch.sum(plans[2,3] * costs[2,3])
                
                
                loss = losses[0,2] + losses[0,3] + losses[1,2] + losses[1,3] - 2 * losses[0,1] - 2 * losses[2,3]
                
                loss.backward()
                c_optimizer.step()
                g_optimizer.step()
                
                loss_to_display.append(float(loss.detach().numpy()))
                
                
            if epoch % print_interval == 0:
                print("Epoch %s: Loss %s;  time (%s)" %
                      (epoch, np.sum(loss_to_display), time()-t))
#                
            if epoch % (print_interval*5) == 0:
                samples = self.sample_data(3)*256.
                for img in samples:
                    plt.figure()
                    plt.imshow(img[:,:,0], cmap='gray')
                    plt.show()
  





(X_train, labels), (_, _) = mnist.load_data()

X_train = X_train / 256.

if(X_train.ndim ==3):
    X_train = np.expand_dims(X_train, axis=3)

img_shape = X_train[0].shape
img_size = img_shape[0] * img_shape[1] * img_shape[2]

critic_dim = 32

gen_params = (32, 256, 512, img_size, img_shape)
disc_params = (img_size, 512, 256, critic_dim)

gan = GAN(X_train, gen_params, disc_params)   

gan.train(20)

samples = gan.sample_data(10)*256.

for img in samples:
    plt.figure()
    plt.imshow(img[:,:,0], cmap='gray')
    plt.show()