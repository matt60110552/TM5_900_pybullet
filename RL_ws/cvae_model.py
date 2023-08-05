import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import ast


# Define the architecture for the encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=256):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        mean = self.mean(x)
        logvar = self.logvar(x)
        return mean, logvar


# Define the architecture for the decoder
# Be careful, the input_dim here means output_dim for decoder
class Decoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=2048):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, input_dim)
        self.act = nn.LeakyReLU()

    def forward(self, z):
        # x = torch.cat([z, x], dim=1)
        z = self.act(self.fc1(z))
        z = self.act(self.fc2(z))
        z = self.act(self.fc3(z))
        x_hat = self.out(z)
        return x_hat


# Define the CVAE
class CVAE(nn.Module):
    def __init__(self, input_dim=9, latent_dim=6, hidden_dim=2048):
        super(CVAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim, hidden_dim)
        self.decoder = Decoder(input_dim, latent_dim, hidden_dim)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        x_recon = self.decoder(z)
        return x_recon, mean, logvar
