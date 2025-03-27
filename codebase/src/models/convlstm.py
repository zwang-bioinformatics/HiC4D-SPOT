# Author: Bishal Shrestha
# Date: 03-24-2025  
# Description: ConvLSTM model architecture

from .convlstmcell import *

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=False, bias=True, return_all_layers=False, dropout=0.3, device='cpu'):
        super(ConvLSTM, self).__init__()

        # self._check_kernel_size_consistency(kernel_size)
        
        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError("Inconsistent list length.")
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.dropout = dropout
        self.device = device
        
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias,
                                          dropout=self.dropout,
                                          device=self.device))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        
        b, _, _, h, w = input_tensor.size()

        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w)) # Shape of hidden state: [(h, c), (h, c), ...] for each layer, where h and c have shape (b, c, h, w)
        
        seq_len = input_tensor.size(1)
        
        cur_layer_input = input_tensor
        
        hs, cs = [], [] # Exected shape: [[(b, c, h, w), (b, c, h, w), ...] for each layer, [(b, c, h, w), (b, c, h, w), ...] for each layer, ...] for each time point
        h_output = []
        
        # go through each time point and then for each timepoint, process LSTM layers
        for t in range(seq_len):    # iterate over the time steps
            
            # Current input tensor at time t
            xt = cur_layer_input[:, t, :, :, :] # Shape of xt: (b, c, h, w)
            residual = xt
            
            # Go through LSTM layers
            hs_t, cs_t = [], [] # List of hidden states and cell states for each layer at time t
            for layer_idx in range(self.num_layers):
                
                # Get hidden and cell states for current layer
                h, c = hidden_state[layer_idx]
                
                # Process current layer
                if layer_idx == 0 and t == 0:
                    # First layer
                    h, c = self.cell_list[layer_idx](input_tensor=xt,
                                                     pre_state=[h, c])
                elif layer_idx == 0 and t > 0:
                    h, c = self.cell_list[layer_idx](input_tensor=h_residual,
                                                     pre_state=[h, c])
                else:
                    # Other layers
                    h, c = self.cell_list[layer_idx](input_tensor=hs_t[layer_idx-1],
                                                     pre_state=[h, c])
                
                hs_t.append(h)
                cs_t.append(c)
                
                # Update hidden state for the next time point
                hidden_state[layer_idx] = [h, c]
                
            # Append hidden states for current time point
            hs.append(hs_t)
            cs.append(cs_t)
            
            # Residual Connection for next time point input
            h_residual = hs_t[self.num_layers-1] + residual
            
            # Append output of last layer
            h_output.append(hs_t[self.num_layers-1])
            
        # prepare h_output for return
        h_output = torch.stack(h_output, dim=1) # Shape of h_output: (b, t, c, h, w)
        
        states_output = [hs, cs]
        
        return h_output, states_output
    
    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    # @staticmethod
    # def _check_kernel_size_consistency(kernel_size):
    #     if not (isinstance(kernel_size, tuple) or
    #             (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
    #         raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
    