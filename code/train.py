import argparse
import torch
import pandas as pd
import torch.utils
import torch.utils.data

import torch.utils.data.dataloader 
from model.RdpGan import RdpGan
from utils.dataset import AdultDataset, CustomMNIST
import utils.logger
import utils.preprocess


torch.manual_seed(28041997)
# Set up argument parser, most of hyperparameters are set here
parser = argparse.ArgumentParser(description='Training script for training RDP-GAN')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--p_bound', type=int, default=6, help='Parameter bound for add_to_weights privacy approch')
parser.add_argument('--latent_dim', type=int, default=100, help='Latent dimension(z) typically [50-200]')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for training')
parser.add_argument('--save_interval', type=int, default=1, help='Interval for saving the model/metrics')
parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'adult'], help='Dataset to train on')
parser.add_argument('--privacy_mode', type=str, default='with_privacy', choices=['with_privacy', 'without_privacy'], help='RDP-GAN Privacy mode')
args = parser.parse_args()




#2 Prepocessing
dataloader, num_features = utils.preprocess.load_dataset(args.dataset, args.batch_size, shuffle=True)





#3 The following parameters should not be changed
utils.logger.data_save_interval = args.save_interval
utils.logger.model_save_interval = args.save_interval
utils.logger.show_log_interval = 1
utils.logger.privacy_mode = args.privacy_mode
utils.logger.num_epochs = args.epochs
utils.logger.dataset_name = args.dataset


#4 Build RDP-GAN  
sigma_scale = [0.1, 0.01, 0.001]
for sigma in sigma_scale:
    rdpGan = RdpGan(input_dim=num_features, output_dim=num_features, latent_dim=args.latent_dim)
    g_losses, d_losses = [], []
    #5 Train RDP-GAN
    for epoch in range(args.epochs):
        rdpGan.train() # Activate training mode 
        for batch_idx, batch_data in enumerate(dataloader):
            current_batch_size = len(batch_data)
            # Train discriminator
            if args.privacy_mode == 'without_privacy':
                d_loss, real_scores, fake_scores = rdpGan.train_d_without_privacy(batch_data)        
            elif args.privacy_mode == 'with_privacy':
                d_loss, real_scores, fake_scores = rdpGan.train_d_with_privacy(batch_data,sigma=sigma, C=args.p_bound)
            # Train generator
            g_loss, fake_data = rdpGan.train_generator(batch_size=current_batch_size)

        # Loggin    
        utils.logger.save_model(epoch, rdpGan.generator)
        utils.logger.log(epoch, d_loss, g_loss, real_scores, fake_scores)        
        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())
        if args.dataset == 'mnist': utils.logger.save_images(epoch, fake_data, sigma)
        utils.logger.save_csv(epoch, data=fake_data.detach().numpy())
        utils.logger.losses_over_epoches(g_losses, d_losses, epoch, sigma)
    if args.privacy_mode == 'without_privacy': break # No need repeat training
