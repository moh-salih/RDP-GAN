import argparse
import torch
import pandas as pd
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader 
import os

from model.RdpGan import RdpGan

from utils.dataset import AdultDataset, CustomMNIST

import utils.logger
import matplotlib.pyplot as plt


# Set up argument parser
parser = argparse.ArgumentParser(description='Training script for training RDP-GAN')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--p_bound', type=int, default=6, help='Parameter bound for add_to_weights privacy approch')
parser.add_argument('--latent_dim', type=int, default=100, help='Latent dimension(z) typically [50-200]')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for training')
parser.add_argument('--save_interval', type=int, default=1, help='Interval for saving the model/metrics')
parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'adult'], help='Dataset to train on')
parser.add_argument('--privacy_mode', type=str, default='no_privacy', choices=['no_privacy', 'add_to_loss', 'add_to_weights'], help='RDP-GAN Privacy mode')

args = parser.parse_args()

# Make use of Cuda in case it's available on your machine. On mine it's not!  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Decide which dataset to train on
if args.dataset == 'mnist':
    dataset = CustomMNIST()
elif args.dataset == 'adult':
    dataset = AdultDataset()

# dataset = MyDataset()
# dataset = CustomMNIST()

#1 Set required hyperparameters
input_dim  = output_dim = dataset.num_features
latent_dim = args.latent_dim
batch_size = args.batch_size
epochs = args.epochs
shuffleDataset = True

#2 Load, Preprocess, and Create required directories
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffleDataset)


#3 Set required parameters to use `utils` properly
utils.logger.data_save_interval = args.save_interval
utils.logger.model_save_interval = args.save_interval
utils.logger.show_log_interval = 1
utils.logger.privacy_mode = args.privacy_mode
# The following parameters shouldn't be changed
utils.logger.num_epochs = epochs
utils.logger.dataset_name = dataset.name


#4 Build RDP-GAN  
sigma_scale = [0, 0.1, 0.01, 0.001]
for sigma in sigma_scale:
    rdpGan = RdpGan(input_dim=input_dim, output_dim=output_dim, latent_dim=latent_dim)
    g_losses, d_losses = [], []

    #5 Train RDP-GAN
    for epoch in range(epochs):

        for batch_idx, batch_data in enumerate(dataloader):
            current_batch_size = len(batch_data)
            
            # Train discriminator
            if args.privacy_mode == 'no_privacy':
                d_loss, real_scores, fake_scores = rdpGan.train_d_with_no_privacy(batch_data)
            
            elif args.privacy_mode == 'add_to_loss':
                d_loss, real_scores, fake_scores = rdpGan.train_d_with_noise_to_loss_privacy(batch_data,sigma=sigma)

            elif args.privacy_mode == 'add_to_weights':
                d_loss, real_scores, fake_scores = rdpGan.train_d_with_noise_to_weights_privacy(batch_data,sigma=sigma, C=args.p_bound)

            # Train generator
            g_loss, fake_data = rdpGan.train_generator(batch_size=current_batch_size)

        utils.logger.save_model(epoch, rdpGan.generator)
        utils.logger.save_csv(epoch, data=fake_data.detach().numpy())
        utils.logger.log(epoch, d_loss, g_loss, real_scores, fake_scores)
        
        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())

        utils.logger.save_images(epoch, fake_data, sigma)
        if epoch % utils.logger.data_save_interval == 0: 
            utils.logger.losses_over_epoches(g_losses, d_losses, epoch, sigma)
