import torch
from model.Generator import Generator
import pandas as pd
import numpy as np
from utils.config import config
import os 

from sklearn.preprocessing import StandardScaler

output_dim=12
latent_dim = 100
weights_path = '../data/output/models/21_05_2024/adult/generator_8.pth'

# Generate random noise
num_samples = 1000  # Number of fake samples to generate
latent_dim = 100  # Dimensionality of the latent space
random_noise = torch.randn(num_samples, latent_dim)  # Generate random noise samples



# Load the generator model
generator = Generator(latent_dim=latent_dim, output_dim=output_dim)  # Instantiate your generator class
generator.load_state_dict(torch.load(weights_path))  # Load the saved model state dict
generator.eval()  # Set the generator to evaluation mode

def rescale(data, min_value=1, max_value=100):
    data_min = -1
    data_max = 1
    return (data - data_min) / (data_max - data_min) * (max_value - min_value) + min_value


# Generate fake data
with torch.no_grad():
    generated_data = generator(random_noise)  # Pass the random noise through the generator

# Post-process the generated data (if necessary)
# For example, if the generated data is in the range [-1, 1], you may want to rescale it to the original range

scaler = StandardScaler()
generated_data = scaler.inverse_transform(generated_data.detach().numpy())
        

# Convert the generated data tensor to a NumPy array (if necessary)
generated_data = generated_data.cpu().numpy()  # Convert to NumPy array if the data is on GPU


# Now you can use the generated data for your application
generated_df = pd.DataFrame(generated_data)


os.makedirs('../data/output/data/21_05_2024/adult', exist_ok=True)
# Save the DataFrame to a CSV file
generated_df.to_csv('../data/output/data/21_05_2024/adult/generated_8.csv', index=False, header=None)