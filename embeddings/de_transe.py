import torch
import torch.nn as nn
import torch.nn.functional as Functional


class DETransE(nn.Module):
    def __init__(self, params):
        super(DETransE, self).__init__()
        self.params = params
        self.name = "DE_TransE"
        self.dataset = params.dataset

        self.learning_rate = 0.001
        self.entity_emb_dim = 90
        self.time_emb_dim = 10
        self.dropout_probability = 0.4

        self.time_nl = torch.sin
        self.sigm = torch.nn.Sigmoid()
        self.tanh = nn.Tanh()

        # Entity and relation embeddings

        self.entity_embeddings = nn.Embedding(self.dataset.num_of_entities(), self.entity_emb_dim).to(self.params.device)
        self.relation_embeddings = nn.Embedding(self.dataset.num_of_relations(), self.entity_emb_dim + self.time_emb_dim).to(self.params.device)

        nn.init.uniform_(self.entity_embeddings.weight)
        nn.init.uniform_(self.relation_embeddings.weight)

        # Time embeddings

        self.year_freq = nn.Embedding(self.dataset.num_of_entities(), self.time_emb_dim).to(self.params.device)
        self.month_freq = nn.Embedding(self.dataset.num_of_entities(), self.time_emb_dim).to(self.params.device)
        self.day_freq = nn.Embedding(self.dataset.num_of_entities(), self.time_emb_dim).to(self.params.device)

        nn.init.uniform_(self.year_freq.weight)
        nn.init.uniform_(self.month_freq.weight)
        nn.init.uniform_(self.day_freq.weight)

        self.year_phi = nn.Embedding(self.dataset.num_of_entities(), self.time_emb_dim).to(self.params.device)
        self.month_phi = nn.Embedding(self.dataset.num_of_entities(), self.time_emb_dim).to(self.params.device)
        self.day_phi = nn.Embedding(self.dataset.num_of_entities(), self.time_emb_dim).to(self.params.device)

        nn.init.uniform_(self.year_phi.weight)
        nn.init.uniform_(self.month_phi.weight)
        nn.init.uniform_(self.day_phi.weight)

        self.year_amp = nn.Embedding(self.dataset.num_of_entities(), self.time_emb_dim).to(self.params.device)
        self.month_amp = nn.Embedding(self.dataset.num_of_entities(), self.time_emb_dim).to(self.params.device)
        self.day_amp = nn.Embedding(self.dataset.num_of_entities(), self.time_emb_dim).to(self.params.device)

        nn.init.uniform_(self.year_amp.weight)
        nn.init.uniform_(self.month_amp.weight)
        nn.init.uniform_(self.day_amp.weight)

    def get_time_embeddings(self, entities, year, month, day):

        y = self.year_amp(entities)*self.time_nl(self.year_freq(entities)*year + self.year_phi(entities))
        m = self.month_amp(entities)*self.time_nl(self.month_freq(entities)*month + self.month_phi(entities))
        d = self.day_amp(entities)*self.time_nl(self.day_freq(entities)*day + self.day_phi(entities))

        return y+m+d

    def get_embeddings(self, heads, rels, tails, years, months, days):
        years = years.view(-1, 1)
        months = months.view(-1, 1)
        days = days.view(-1, 1)

        h, r, t = self.entity_embeddings(heads), self.relation_embeddings(rels), self.entity_embeddings(tails)

        h_t = self.get_time_embeddings(heads, years, months, days)
        t_t = self.get_time_embeddings(tails, years, months, days)

        h = torch.cat((h, h_t), 1)
        t = torch.cat((t, t_t), 1)
        return h, r, t

    def forward(self, heads, rels, tails, years, months, days):
        head_emb, relation_emb, tail_emb = self.get_embeddings(heads, rels, tails, years, months, days)

        scores = self.scoring_function(head_emb, relation_emb, tail_emb)
        scores = Functional.dropout(scores, p=self.params.dropout, training=self.training)

        return scores

    def scoring_function(self, head_emb, relation_emb, tail_emb):
        scores = torch.add(head_emb, relation_emb)
        scores = torch.sub(scores, tail_emb)
        scores = torch.linalg.norm(scores, dim=1)

        return scores

    def loss_function(self, score, target):
        return (torch.sum(score) - torch.sum(target)).abs() / score.shape[0]

