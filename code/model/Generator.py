from torch import nn


class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.output_dim = output_dim


        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, output_dim),
            nn.Tanh() 
        )
        
    def forward(self, x):
        return self.model(x)
