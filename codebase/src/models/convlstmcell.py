# Author: Bishal Shrestha
# Date: 03-24-2025  
# Description: ConvLSTM cell architecture

import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias, dropout, device):
        super(ConvLSTMCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim # Number of channels in the hidden state of the ConvLSTM cell
        self.kernel_size = kernel_size	
        self.padding = kernel_size // 2	# Padding is half of the kernel size to maintain the input size
        self.bias = bias
        self.dropout = dropout
        self.device = device

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,	# Because in ConvLSTMCell, the input tensor is concatenated with the hidden state tensor
            out_channels=4 * self.hidden_dim,	# 4 gates: input, forget, output, and cell state
            kernel_size=self.kernel_size,	# Kernel size of the convolutional layer
            padding=self.padding,	
            bias=self.bias
        )
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, input_tensor, pre_state):
        # input_tensor of shape (b, c, h, w) and pre_state of shape [(b, c, h, w), (b, c, h, w)]
        h_pre, c_pre = pre_state
        
        combined = torch.cat([input_tensor, h_pre], dim=1)  # concatenate along channel axis. Shape: (b, c_in + c_hid, h, w)
        combined_conv_outputs = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv_outputs, self.hidden_dim, dim=1)
        
        # here i is input gate, f is forget gate, o is output gate, g is cell state
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        # cell state update
        c = f * c_pre + i * g
        h = o * torch.tanh(c)
        
        h = self.dropout_layer(h)
        
        return h, c

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.device))
