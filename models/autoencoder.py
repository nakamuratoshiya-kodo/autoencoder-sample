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
    def __init__(self, input_shape, latent_dim, hidden_dims, activation="ReLU",normalize=True):
        super(ConvAutoencoder, self).__init__()
        channels, height, width = input_shape
        act_fn = get_activation(activation)

        # -------------------------------
        # エンコーダ構築 (hidden_dims参照)
        # -------------------------------
        encoder_layers = []
        in_channels = channels
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1))
            encoder_layers.append(act_fn)
            in_channels = h_dim
        self.encoder = nn.Sequential(*encoder_layers, nn.Flatten())

        # 潜在空間
        self.flatten_dim = (height // 2 ** len(hidden_dims)) * (width // 2 ** len(hidden_dims)) * hidden_dims[-1]
        self.latent = nn.Sequential(
            nn.Linear(self.flatten_dim, latent_dim),
            act_fn,
            nn.Linear(latent_dim, self.flatten_dim),
            act_fn
        )

        # -------------------------------
        # デコーダ構築 (hidden_dimsの逆順)
        # -------------------------------
        decoder_layers = []
        out_channels_list = list(reversed(hidden_dims))
        self.decoder_input = nn.Unflatten(1, (out_channels_list[0], height // 2 ** len(hidden_dims), width // 2 ** len(hidden_dims)))
        in_channels = out_channels_list[0]
        for h_dim in out_channels_list[1:]:
            decoder_layers.append(nn.ConvTranspose2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1, output_padding=1))
            decoder_layers.append(act_fn)
            in_channels = h_dim

        # 最終出力層
        decoder_layers.append(nn.ConvTranspose2d(in_channels, channels, kernel_size=3, stride=2, padding=1, output_padding=1))
        
        if normalize:
            decoder_layers.append(nn.Tanh())  # 正規化を行う場合はTanhを使用
        else:
            decoder_layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        z = self.latent(z)
        z = self.decoder_input(z)
        x_recon = self.decoder(z)
        return x_recon
