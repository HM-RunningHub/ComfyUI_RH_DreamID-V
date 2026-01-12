import math
import torch
import torch.nn as nn
from einops import rearrange
from typing import List, Tuple
from torch import Tensor
import os
from PIL import Image
import numpy as np

class Projector(nn.Module):
    def __init__(self, cin=14, cout=16, vid_dim=1536, patch_size = [1,2,2] ):
        super(Projector, self).__init__()
        self.time_down = nn.Sequential(
            nn.Conv1d(in_channels=cin, out_channels=cin, kernel_size=3, stride=2, padding=1),
            nn.Conv1d(in_channels=cin, out_channels=cin, kernel_size=3, stride=2, padding=1)
        )
        self.convs = nn.Sequential(
            nn.Conv2d(cin, 128, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(512, cout, kernel_size=3,padding=1),
        )

        
    def forward(self, x):
 
        x = rearrange(x, "b c f h w -> (b f) c h w")
        h, w = x.shape[-2:]
        x = rearrange(x, 'f c h w -> (h w) c f')
        x = self.time_down(x)
        x = rearrange(x, '(h w) c f -> f c h w', h=h, w=w)

        x = self.convs(x)
        x = rearrange(x, 't c h w -> c t h w')
        return x, x.shape
if __name__ == '__main__':
    input_tensor = torch.randn(1, 3, 77, 512, 512)
    projector = Projector(cin=3)
    output_tensor, output_shape = projector(input_tensor)
    print(output_tensor.shape)

    #torch.Size([81920, 1536])
    #tensor([20, 64, 64])