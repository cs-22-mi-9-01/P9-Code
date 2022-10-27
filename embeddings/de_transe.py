import torch.nn as nn


class DETransE(nn.Module):
    def __init__(self, params):
        self.embedding = params.dataset

    def

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
