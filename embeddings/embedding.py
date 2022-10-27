import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, args):
        super(Embedding, self).__init__()

    def create_embeddings(self):
        raise NotImplementedError


