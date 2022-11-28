import torch
from torch import nn
import torch.nn.functional as F


L = 1.0

def convert_to_logic(x):
    # [0, 1]
    y = torch.sigmoid(2 * x)
    return y


def convert_to_feature(x):
    # [-1, 1]
    y = torch.tanh(x)
    return y


def convert_to_time_feature(x):
    # [-1, 1]
    y = torch.tanh(x)
    return y


def convert_to_time_logic(x):
    # [0, 1]
    y = torch.sigmoid(2 * x)
    return y


def scale_feature(feature):
    # f,f' in [-L, L]
    # f' = (f + 2L) % (2L) - L, where L=1
    indicator_positive = feature >= 0
    indicator_negative = feature < 0
    feature[indicator_positive] = feature[indicator_positive] - L
    feature[indicator_negative] = feature[indicator_negative] + L
    return feature

class EntityProjection(nn.Module):
    def __init__(self, dim, hidden_dim=800, num_layers=2, drop=0.1):
        super(EntityProjection, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(drop)
        token_dim = dim * 4
        self.layer1 = nn.Linear(token_dim, self.hidden_dim)
        self.layer0 = nn.Linear(self.hidden_dim, token_dim)
        for i in range(2, num_layers + 1):
            setattr(self, f"layer{i}", nn.Linear(self.hidden_dim, self.hidden_dim))
        for i in range(num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, f"layer{i}").weight)

    def forward(self,
                q_feature, q_logic, q_time_feature, q_time_logic,
                r_feature, r_logic, r_time_feature, r_time_logic,
                t_feature, t_logic, t_time_feature, t_time_logic):
        x = torch.cat([
            q_feature + r_feature + t_feature,
            q_logic + r_logic + t_logic,
            q_time_feature + r_time_feature + t_time_feature,
            q_time_logic + r_time_logic + t_time_logic,
        ], dim=-1)
        for i in range(1, self.num_layers + 1):
            x = F.relu(getattr(self, f"layer{i}")(x))
        x = self.layer0(x)

        feature, logic, time_feature, time_logic = torch.chunk(x, 4, dim=-1)
        feature = convert_to_feature(feature)
        logic = convert_to_logic(logic)
        time_feature = convert_to_time_feature(time_feature)
        time_logic = convert_to_time_logic(time_logic)
        return feature, logic, time_feature, time_logic



class EntityIntersection(nn.Module):
    def __init__(self, dim):
        super(EntityIntersection, self).__init__()
        self.dim = dim
        self.feature_layer_1 = nn.Linear(self.dim * 2, self.dim)
        self.feature_layer_2 = nn.Linear(self.dim, self.dim)
        self.time_feature_layer_1 = nn.Linear(self.dim * 2, self.dim)
        self.time_feature_layer_2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.feature_layer_1.weight)
        nn.init.xavier_uniform_(self.feature_layer_2.weight)
        nn.init.xavier_uniform_(self.time_feature_layer_1.weight)
        nn.init.xavier_uniform_(self.time_feature_layer_2.weight)

    def forward(self, feature, logic, time_feature, time_logic):
        # N x B x d
        logits = torch.cat([feature, logic], dim=-1)  # N x B x 2d
        feature_attention = F.softmax(self.feature_layer_2(F.relu(self.feature_layer_1(logits))), dim=0)
        feature = torch.sum(feature_attention * feature, dim=0)

        logits = torch.cat([time_feature, time_logic], dim=-1)  # N x B x 2d
        feature_attention = F.softmax(self.time_feature_layer_2(F.relu(self.time_feature_layer_1(logits))), dim=0)
        time_feature = torch.sum(feature_attention * time_feature, dim=0)

        logic, _ = torch.min(logic, dim=0)
        time_logic, _ = torch.min(time_logic, dim=0)
        return feature, logic, time_feature, time_logic

    

class EntityUnion(nn.Module):
    def __init__(self, dim):
        super(EntityUnion, self).__init__()
        self.dim = dim
        self.feature_layer_1 = nn.Linear(self.dim * 2, self.dim)
        self.feature_layer_2 = nn.Linear(self.dim, self.dim)
        self.time_feature_layer_1 = nn.Linear(self.dim * 2, self.dim)
        self.time_feature_layer_2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.feature_layer_1.weight)
        nn.init.xavier_uniform_(self.feature_layer_2.weight)
        nn.init.xavier_uniform_(self.time_feature_layer_1.weight)
        nn.init.xavier_uniform_(self.time_feature_layer_2.weight)

    def forward(self, feature, logic, time_feature, time_logic):
        # N x B x d
        logits = torch.cat([feature, logic], dim=-1)  # N x B x 2d
        feature_attention = F.softmax(self.feature_layer_2(F.relu(self.feature_layer_1(logits))), dim=0)
        feature = torch.sum(feature_attention * feature, dim=0)

        logits = torch.cat([time_feature, time_logic], dim=-1)  # N x B x 2d
        feature_attention = F.softmax(self.time_feature_layer_2(F.relu(self.time_feature_layer_1(logits))), dim=0)
        time_feature = torch.sum(feature_attention * time_feature, dim=0)

        logic, _ = torch.max(logic, dim=0)
        # for time, it is intersection
        time_logic, _ = torch.min(time_logic, dim=0)
        # logic = torch.prod(logic, dim=0)
        return feature, logic, time_feature, time_logic



class EntityNegation(nn.Module):
    def __init__(self, dim):
        super(EntityNegation, self).__init__()
        self.dim = dim
        self.feature_layer_1 = nn.Linear(self.dim * 2, self.dim)
        self.feature_layer_2 = nn.Linear(self.dim, self.dim)
        nn.init.xavier_uniform_(self.feature_layer_1.weight)
        nn.init.xavier_uniform_(self.feature_layer_2.weight)

    def forward(self, feature, logic, time_feature, time_logic):
        logits = torch.cat([feature, logic], dim=-1)  # N x B x 2d
        feature = self.feature_layer_2(F.relu(self.feature_layer_1(logits)))
        logic = 1 - logic
        return feature, logic, time_feature, time_logic



class TimeProjection(nn.Module):
    def __init__(self, dim, hidden_dim=800, num_layers=2, drop=0.1):
        super(TimeProjection, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(drop)
        token_dim = dim * 4
        self.layer1 = nn.Linear(token_dim, self.hidden_dim)
        self.layer0 = nn.Linear(self.hidden_dim, token_dim)
        for nl in range(2, num_layers + 1):
            setattr(self, "layer{}".format(nl), nn.Linear(self.hidden_dim, self.hidden_dim))
        for nl in range(num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, "layer{}".format(nl)).weight)

    def forward(self,
                q1_feature, q1_logic, q1_time_feature, q1_time_logic,
                r_feature, r_logic, r_time_feature, r_time_logic,
                q2_feature, q2_logic, q2_time_feature, q2_time_logic):
        x = torch.cat([
            q1_feature + r_feature + q2_feature,
            q1_logic + r_logic + q2_logic,
            q1_time_feature + r_time_feature + q2_time_feature,
            q1_time_logic + r_time_logic + q2_time_logic,
        ], dim=-1)
        for nl in range(1, self.num_layers + 1):
            x = F.relu(getattr(self, "layer{}".format(nl))(x))
        x = self.layer0(x)

        feature, logic, time_feature, time_logic = torch.chunk(x, 4, dim=-1)
        feature = convert_to_feature(feature)
        logic = convert_to_logic(logic)
        time_feature = convert_to_time_feature(time_feature)
        time_logic = convert_to_time_logic(time_logic)
        return feature, logic, time_feature, time_logic

class TemporalIntersection(nn.Module):
    def __init__(self, dim):
        super(TemporalIntersection, self).__init__()
        self.dim = dim
        self.feature_layer_1 = nn.Linear(self.dim * 2, self.dim)
        self.feature_layer_2 = nn.Linear(self.dim, self.dim)
        self.time_feature_layer_1 = nn.Linear(self.dim * 2, self.dim)
        self.time_feature_layer_2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.feature_layer_1.weight)
        nn.init.xavier_uniform_(self.feature_layer_2.weight)
        nn.init.xavier_uniform_(self.time_feature_layer_1.weight)
        nn.init.xavier_uniform_(self.time_feature_layer_2.weight)

    def forward(self, feature, logic, time_feature, time_logic):
        # N x B x d
        logits = torch.cat([feature, logic], dim=-1)  # N x B x 2d
        feature_attention = F.softmax(self.feature_layer_2(F.relu(self.feature_layer_1(logits))), dim=0)
        feature = torch.sum(feature_attention * feature, dim=0)

        logits = torch.cat([time_feature, time_logic], dim=-1)  # N x B x 2d
        feature_attention = F.softmax(self.time_feature_layer_2(F.relu(self.time_feature_layer_1(logits))), dim=0)
        time_feature = torch.sum(feature_attention * time_feature, dim=0)

        logic, _ = torch.min(logic, dim=0)
        time_logic, _ = torch.min(time_logic, dim=0)
        return feature, logic, time_feature, time_logic
    
class TemporalUnion(nn.Module):
    def __init__(self, dim):
        super(TemporalUnion, self).__init__()
        self.dim = dim
        self.feature_layer_1 = nn.Linear(self.dim * 2, self.dim)
        self.feature_layer_2 = nn.Linear(self.dim, self.dim)
        self.time_feature_layer_1 = nn.Linear(self.dim * 2, self.dim)
        self.time_feature_layer_2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.feature_layer_1.weight)
        nn.init.xavier_uniform_(self.feature_layer_2.weight)
        nn.init.xavier_uniform_(self.time_feature_layer_1.weight)
        nn.init.xavier_uniform_(self.time_feature_layer_2.weight)

    def forward(self, feature, logic, time_feature, time_logic):
        # N x B x d
        logits = torch.cat([feature, logic], dim=-1)  # N x B x 2d
        feature_attention = F.softmax(self.feature_layer_2(F.relu(self.feature_layer_1(logits))), dim=0)
        feature = torch.sum(feature_attention * feature, dim=0)

        logits = torch.cat([time_feature, time_logic], dim=-1)  # N x B x 2d
        feature_attention = F.softmax(self.time_feature_layer_2(F.relu(self.time_feature_layer_1(logits))), dim=0)
        time_feature = torch.sum(feature_attention * time_feature, dim=0)

        # for entity, it is intersection
        logic, _ = torch.min(logic, dim=0)
        # for time, it is union
        time_logic, _ = torch.max(time_logic, dim=0)
        # logic = torch.prod(logic, dim=0)
        return feature, logic, time_feature, time_logic

class TemporalNegation(nn.Module):
    def __init__(self, dim):
        super(TemporalNegation, self).__init__()
        self.dim = dim
        self.feature_layer_1 = nn.Linear(self.dim * 2, self.dim)
        self.feature_layer_2 = nn.Linear(self.dim, self.dim)
        nn.init.xavier_uniform_(self.feature_layer_1.weight)
        nn.init.xavier_uniform_(self.feature_layer_2.weight)

    def forward(self, feature, logic, time_feature, time_logic):
        logits = torch.cat([time_feature, time_logic], dim=-1)  # N x B x 2d
        time_feature = self.feature_layer_2(F.relu(self.feature_layer_1(logits)))
        time_logic = 1 - time_logic
        return feature, logic, time_feature, time_logic

    

class TemporalBefore(nn.Module):
    def __init__(self, dim):
        super(TemporalBefore, self).__init__()
        self.dim = dim

    def forward(self, feature, logic, time_feature, time_logic):
        time_feature = scale_feature(time_feature - L / 2 - time_logic / 2)
        time_logic = (L - time_logic) / 2

        return feature, logic, time_feature, time_logic


class TemporalAfter(nn.Module):
    def __init__(self, dim):
        super(TemporalAfter, self).__init__()
        self.dim = dim

    def forward(self, feature, logic, time_feature, time_logic):
        time_feature = scale_feature(time_feature + L / 2 + time_logic / 2)
        time_logic = (L - time_logic) / 2

        return feature, logic, time_feature, time_logic


class TemporalNext(nn.Module):
    def __init__(self):
        super(TemporalNext, self).__init__()

    def forward(self, feature, logic, time_feature, time_logic):
        time_feature = scale_feature(time_feature)
        time_logic = 1 - time_logic
        return feature, logic, time_feature, time_logic