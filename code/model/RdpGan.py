import torch
from model.Discriminator import Discriminator
from model.Generator import Generator
import os, copy
import numpy as np


class RdpGan():
    def __init__(self, input_dim, output_dim, latent_dim, device='cpu', d_lr=0.0003, g_lr=0.0003):
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.d_lr = d_lr
        self.g_lr = g_lr
        

        # Define models
        self.generator = Generator(output_dim=self.output_dim, latent_dim=self.latent_dim)
        self.discriminator = Discriminator(input_dim=self.input_dim)

        # Criterion
        self.criterion = torch.nn.BCELoss()        
        
        # Optimizers
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.g_lr) 
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.d_lr)
       
    def train_generator(self, batch_size):
        real_label = torch.ones((batch_size, 1))
        noise = torch.randn(batch_size, self.latent_dim)
        fake_data = self.generator(noise)
        

        output = self.discriminator(fake_data)
        g_loss = self.criterion(output, real_label)
        
        # backpropogate and optimize
        self.g_optimizer.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()

        return g_loss, fake_data

    def train_d_with_no_privacy(self, real_data):
        batch_size = len(real_data)
        
        real_label = torch.ones((batch_size, 1))
        fake_label = torch.zeros((batch_size, 1))

        # For Real Data
        real_out = self.discriminator(real_data)
        real_scores = real_out

        # For Fake Data
        noise = torch.randn(batch_size, self.latent_dim)
        fake_data = self.generator(noise)
        fake_out = self.discriminator(fake_data)
        fake_scores = fake_out
        
        # Compute loss
        d_real_loss = self.criterion(real_out, real_label)
        d_fake_loss = self.criterion(fake_out, fake_label)
        d_loss = d_real_loss + d_fake_loss

        # Optimize and backpropogate
        self.d_optimizer.zero_grad()
        d_loss.backward()
        self.d_optimizer.step()

        return d_loss, real_scores, fake_scores

    def train_d_with_noise_to_loss_privacy(self, real_data, sigma):
        batch_size = len(real_data)
        
        real_label = torch.ones((batch_size, 1))
        fake_label = torch.zeros((batch_size, 1))

        # For Real Data
        real_out = self.discriminator(real_data)
        real_scores = real_out

        # For Fake Data
        noise = torch.randn(batch_size, self.latent_dim)
        fake_data = self.generator(noise)
        fake_out = self.discriminator(fake_data)
        fake_scores = fake_out
        
        # Compute loss
        d_real_loss = self.criterion(real_out, real_label)
        d_fake_loss = self.criterion(fake_out, fake_label)
        d_loss = d_real_loss + d_fake_loss
        d_loss += np.random.normal(0, sigma)

        # Optimize and backpropogate
        self.d_optimizer.zero_grad()
        d_loss.backward()
        self.d_optimizer.step()

        return d_loss, real_scores, fake_scores

    def train_d_with_noise_to_weights_privacy(self, real_data, sigma, C=6):
        batch_size = len(real_data)
        
        real_label = torch.ones((batch_size, 1))
        fake_label = torch.zeros((batch_size, 1))

        # For Real Data
        real_out = self.discriminator(real_data)
        real_scores = real_out

        # For Fake Data
        noise = torch.randn(batch_size, self.latent_dim)
        fake_data = self.generator(noise)
        fake_out = self.discriminator(fake_data)
        fake_scores = fake_out
        
        # Compute loss
        d_real_loss = self.criterion(real_out, real_label)
        d_fake_loss = self.criterion(fake_out, fake_label)
        d_loss = d_real_loss + d_fake_loss

        # Optimize and backpropogate
        self.d_optimizer.zero_grad()
        d_loss.backward()
        
        # Add noise to parameters
        d_parameters = self.discriminator.state_dict()
        s_noise = copy.deepcopy(d_parameters)
        bound = []
        bound_c = [] 
        j = 0
        for k in d_parameters.keys(): 
            noise = np.random.normal(0, sigma, d_parameters[k].size())
            noise = torch.from_numpy(noise).float()
            s_noise[k] = d_parameters[k] + noise
            bound.append(np.linalg.norm(s_noise[k]))
            if bound[j] > C:
                s_noise[k] = s_noise[k] / bound[j] * C
            bound_c.append(np.linalg.norm(s_noise[k]))
            j +=1
        self.discriminator.load_state_dict(d_parameters)
        

        self.d_optimizer.step()

        return d_loss, real_scores, fake_scores
    
