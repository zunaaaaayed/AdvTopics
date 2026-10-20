
import torch


class ConvNet(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()
