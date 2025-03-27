# Author: Bishal Shrestha
# Date: 03-24-2025  
# Description: HiC4D-SPOT model architecture

from .convlstm import *
import torch.nn.functional as F

class HiC4D_SPOT(nn.Module):
    def __init__(self, input_dim, device, dropout_prob=0.3, lambda_consistency=0.1, lambda_contrastive=0.2, margin=1.0):
        super(HiC4D_SPOT, self).__init__()
        
        self.input_dim = input_dim # 1 (number of channels)
        self.dropout_prob = dropout_prob
        self.device = device
        
        self.loss_mse = nn.MSELoss()
        
        # Spatial encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(self.input_dim, 256, kernel_size=7, stride=2, padding=3),  # 1x50x50 -> 256x25x25
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Conv2d(256, 128, kernel_size=5, stride=2, padding=2),    # 256x25x25 -> 128x13x13
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),   # 128x13x13 -> 64x7x7
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob)
        )

        # Temporal encoder-decoder
        self.temporal_encoder = ConvLSTM(input_dim=64, hidden_dim=[64, 64, 32, 64, 64], kernel_size=3, num_layers=5, batch_first=True, dropout=self.dropout_prob, device=self.device)

        # Spatial decoder
        # Hout = (Hin - 1)*stride - 2*padding + dilation*(kernel_size-1) + output_padding + 1
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=0),  # 64x7x7 -> 128x13x13
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.ConvTranspose2d(128, 256, kernel_size=5, stride=2, padding=2, output_padding=0),  # 128x13x13 -> 256x25x25
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.ConvTranspose2d(256, self.input_dim, kernel_size=7, stride=2, padding=3, output_padding=1),  # 256x25x25 -> 1x50x50
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x_in = x        # Shape: (batch_size, seq_len, channels, height, width)
        batch_size, seq_len, _, h, w = x.size()
        x = x.view(batch_size * seq_len, self.input_dim, h, w)  # Flatten the batch and sequence dimensions for the encoder
        x = self.encoder(x)
        _, _, h_enc, w_enc = x.size()
        x = x.view(batch_size, seq_len, 64, h_enc, w_enc)  # Reshape to (batch_size, seq_len, channels, height, width)
        x, _ = self.temporal_encoder(x)
        x = x.contiguous().view(batch_size * seq_len, 64, h_enc, w_enc)  # Flatten the batch and sequence dimensions for the decoder
        x = self.decoder(x)
        x = x.view(batch_size, seq_len, self.input_dim, h, w)  # Reshape back to (batch_size, seq_len, channels, height, width)
        
        # Compute losses
        loss_mse = self.loss_mse(x, x_in)
        
        # Combined loss
        loss = loss_mse
        max_pixel_loss = torch.max(torch.abs(x - x_in))
        
        return x, loss, max_pixel_loss
    