# models/autoencoder.py
# CNNベースオートエンコーダの定義

import torch
import torch.nn as nn

def get_activation(name):
    activations = {
        "ReLU": nn.ReLU(),
        "LeakyReLU": nn.LeakyReLU(),
        "Sigmoid": nn.Sigmoid(),
        "Tanh": nn.Tanh()
    }
    return activations.get(name, nn.ReLU())

class ConvAutoencoder(nn.Module):
    def __init__(self, input_shape, latent_dim, activation="ReLU"):
        super(ConvAutoencoder, self).__init__()
        channels, height, width = input_shape
        act_fn = get_activation(activation)

        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, stride=2, padding=1),
            act_fn,
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            act_fn,
            nn.Flatten(),
            nn.Linear((height//4)*(width//4)*64, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, (height//4)*(width//4)*64),
            nn.Unflatten(1, (64, height//4, width//4)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            act_fn,
            nn.ConvTranspose2d(32, channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
