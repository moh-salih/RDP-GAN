import torch, os, argparse 
from model.Generator import Generator
import pandas as pd
import numpy as np
import utils.logger
from utils.logger import ROOT_DIR, today


torch.manual_seed(28041997)


# Set up argument parser, most of hyperparameters are set here
parser = argparse.ArgumentParser(description='Training script for training RDP-GAN')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--p_bound', type=int, default=6, help='Parameter bound for add_to_weights privacy approch')
parser.add_argument('--latent_dim', type=int, default=100, help='Latent dimension(z) typically [50-200]')
parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'adult'], help='Dataset to generate fake data similar to')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for training')
parser.add_argument('--privacy_mode', type=str, default='with_privacy', choices=['with_privacy', 'without_privacy'], help='RDP-GAN Privacy mode')
args = parser.parse_args()


# These variables are required for saving generated data properly
utils.logger.dataset_name = args.dataset
utils.logger.privacy_mode = args.privacy_mode

generator_weights = r'../data/working/31_05_2024/mnist/models/without_privacy/epoch_100_sigma_0.pth'
if args.dataset == 'mnist':
    output_dim = 784
elif args.dataset == 'adult':
    output_dim = 12

generator = Generator(args.latent_dim, output_dim)
generator.load_state_dict(torch.load(generator_weights))

with torch.no_grad():
    for i in range(10):
        noise = torch.randn(args.batch_size, args.latent_dim)
        fake_output = generator(noise)
        utils.logger.save_output_images(fake_output) if args.dataset == 'mnist' else utils.logger.save_output_csv(fake_output) 
