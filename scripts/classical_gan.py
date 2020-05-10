#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 16:07:20 2020

@author: slimane
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from keras.datasets import mnist
from time import time


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
        x = torch.tanh(self.map3(x))
        return torch.reshape(x, (-1,)+self.output_shape)

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size_1)
        self.map2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.map3 = nn.Linear(hidden_size_2, 1)
        self.f = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = nn.Flatten()(x)
        x = self.f(self.map1(x))
        x = self.f(self.map2(x))
        return torch.sigmoid(self.map3(x))
    

class GAN():
    
    def __init__(self, X_train, gen_params, disc_params):
        self.X_train = X_train

        # default parameters for mnist 
        self.img_rows = X_train.shape[1]
        self.img_cols = X_train.shape[2]
        self.img_channels = X_train.shape[3]
        self.img_shape = (self.img_rows, self.img_cols, self.img_channels)
        self.z_dim = 32
        
        self.generator = Generator(*gen_params)
        self.discriminator = Discriminator(*disc_params)
        
            
    
    def sample_data(self, n_sample=100):
        z_random = np.random.randn(n_sample, self.z_dim)
        samples = self.generator(torch.FloatTensor(z_random))
        samples = samples.detach().numpy()
        return samples
        
    
    
    def train(self, epochs=1000, batch_size=128, print_interval=1):
        
        criterion = nn.BCELoss()  # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss
        d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0001)
        g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0001)
        d_steps = 1
        g_steps = 1
        
        
        for epoch in range(epochs):
            
            t=time()
            
            for i in range(self.X_train.shape[0]//batch_size):
                idx = np.random.choice(self.X_train.shape[0], size=batch_size, replace=False)
                
                for d_index in range(d_steps):
                    # 1. Train D on real+fake
                    self.discriminator.zero_grad()
        
                    #  1A: Train D on real
                    d_real_data = torch.FloatTensor(X_train[idx])
                    d_real_decision = self.discriminator(d_real_data)
                    d_real_error = criterion(torch.min(d_real_decision, (1. - 1e-8) * torch.ones(d_real_decision.size())), torch.ones((batch_size, 1)))  # ones = true
                    d_real_error.backward() # compute/store gradients, but don't change params
        
                    #  1B: Train D on fake
                    d_gen_input = torch.FloatTensor(np.random.randn(batch_size, self.z_dim))
                    d_fake_data = self.generator(d_gen_input).detach()  # detach to avoid training G on these labels
                    d_fake_decision = self.discriminator(d_fake_data)
                    d_fake_error = criterion(torch.max(d_fake_decision, 1e-8 * torch.ones(d_fake_decision.size())), torch.zeros((batch_size, 1)))  # zeros = fake
                    d_fake_error.backward()
                    d_optimizer.step()     # Only optimizes D's parameters; changes based on stored gradients from backward()
        
                    dre, dfe = d_real_error.detach().numpy(), d_fake_error.detach().numpy()
        
                for g_index in range(g_steps):
                    # 2. Train G on D's response (but DO NOT train D on these labels)
                    self.generator.zero_grad()
        
                    gen_input = torch.FloatTensor(np.random.randn(batch_size, self.z_dim))
                    g_fake_data = self.generator(gen_input)
                    dg_fake_decision = self.discriminator(g_fake_data)
                    g_error = torch.mean(torch.log(1-dg_fake_decision))  # Train G to pretend it's genuine
        
                    g_error.backward()
                    g_optimizer.step()  # Only optimizes G's parameters
                    ge = g_error.detach().numpy()
    
            if epoch % print_interval == 0:
                print("Epoch %s: D (%s real_err, %s fake_err) G (%s err);  time (%s)" %
                      (epoch, dre, dfe, ge, time()-t))
                
            if epoch % (print_interval*5) == 0:
                samples = self.sample_data(3)*0.5 + 0.5
                for img in samples:
                    plt.figure()
                    plt.imshow(img[:,:,0], cmap='gray')
                    plt.show()
  

def load_data(dataset_name):
     # Load the dataset
    if(dataset_name == 'mnist'):
        (X_train, _), (_, _) = mnist.load_data()
    elif(dataset_name == 'cifar'):
        from keras.datasets import cifar10
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    else:
        print('Error, unknown database')

    # Rescale -1 to 1
    X_train = X_train / 127.5 - 1.
    #add a channel dimension, if need be (for mnist data)
    if(X_train.ndim ==3):
        X_train = np.expand_dims(X_train, axis=3)
    return X_train

X_train = load_data('mnist')
      
        
img_shape = X_train[0].shape
img_size = img_shape[0] * img_shape[1] * img_shape[2]

gen_params = (32, 256, 512, img_size, img_shape)
disc_params = (img_size, 512, 256)

gan = GAN(X_train, gen_params, disc_params)       

gan.train(10)

samples = gan.sample_data(10)*0.5 + 0.5

for img in samples:
    plt.figure()
    plt.imshow(img[:,:,0], cmap='gray')
    plt.show()