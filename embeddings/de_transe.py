import torch
import torch.nn as nn
from torch.nn.modules.loss import _WeightedLoss
from torch import Tensor


class DETransE(nn.Module):
    def __init__(self, params):
        super(DETransE, self).__init__()
        self.params = params
        self.learning_rate = 0.001
        self.embedding_dimensions = 3

        self.entity_embedding = nn.Embedding(len(self.params.dataloader.dataset.ids["entity"]), self.embedding_dimensions)
        self.relation_embedding = nn.Embedding(len(self.params.dataloader.dataset.ids["relation"]), self.embedding_dimensions)
        self.time_embedding = nn.Embedding(len(self.params.dataloader.dataset.ids["time"]), self.embedding_dimensions)

    def forward(self, heads, rels, tails, times):
        h_embs, r_embs, t_embs = self.getEmbeddings(heads, rels, tails, years, months, days)

        scores = h_embs + r_embs - t_embs
        scores = F.dropout(scores, p=self.params.dropout, training=self.training)
        scores = -torch.norm(scores, dim=1)
        return scores

