import torch.nn as nn
import torch.nn.functional as F
import torch


class SegNet(nn.Module):
    def __init__(self):
        super().__init__()
