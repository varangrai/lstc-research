__all__ = ['PatchTST_backbone']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
#from collections import OrderedDict
# from layers.PatchTST_layers import *
# from layers.RevIN import RevIN

__all__ = ['Transpose', 'get_activation_fn', 'moving_avg', 'series_decomp', 'PositionalEncoding', 'SinCosPosEncoding', 'Coord2dPosEncoding', 'Coord1dPosEncoding', 'positional_encoding']           

class base_Model(nn.Module):
    def __init__(self, configs, patch_len=32, d_model=128, n_layers=4, n_heads=4, res_attention = True):
        super(base_Model, self).__init__()

        # Convolutional blocks from base_Model
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, 32, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size//2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        model_output_dim = configs.features_len
        self.logits = nn.Linear(model_output_dim * configs.final_out_channels, configs.num_classes)

       

    def calculate_mixed_output_dim(self, configs):
        # Calculate the output size here after ChannelMixing based on your specific sequence length
        # and channel mixing implementation. This is a placeholder function.
        # For example, if your sequence length is halved after each conv block:
        seq_len_after_convs = configs.features_len // (2 * 2 * 2)  # after three max pooling with kernel_size=2
        seq_len_after_patches = (seq_len_after_convs + self.channel_mixing.patch_len - 1) // self.channel_mixing.patch_len
        return seq_len_after_patches * self.channel_mixing.d_model

    def forward(self, x_in): # [128,9,128]
        # Pass through the convolutional blocks

        # Pass through the ChannelMixing module
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x_flat = x.reshape(x.shape[0], -1)

        logits = self.logits(x_flat)
        return logits, x
