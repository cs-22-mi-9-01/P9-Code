import torch
from torch import nn

import TFLEX.expression.TFLEX_DSL as Parser
from TFLEX.expression.TFLEX_DSL import is_to_predict_entity_set
from TFLEX.expression.ParamSchema import is_entity, is_relation, is_timestamp
from TFLEX.modules import EntityProjection, EntityIntersection, EntityUnion, EntityNegation, TimeProjection, TemporalIntersection, TemporalUnion, TemporalNegation, TemporalBefore, TemporalAfter, TemporalNext


L = 1

def convert_to_logic(x):
    # [0, 1]
    y = torch.sigmoid(2 * x)
    return y


def convert_to_feature(x):
    # [-1, 1]
    y = torch.tanh(x) * L
    return y


def convert_to_time_feature(x):
    # [-1, 1]
    y = torch.tanh(x) * L
    return y


def convert_to_time_logic(x):
    # [0, 1]
    y = torch.sigmoid(2 * x)
    return y

class FLEX(nn.Module):
    def __init__(self) -> None:
        super(FLEX, self).__init__()
        nentity = 7128
        nrelation = 460
        hidden_dim = 800
        ntimestamp = 365
        drop = 0.1
        test_batch_size = 1
        gamma = 30.0
        center_reg = 0.02

        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim
        self.timestamp_dim = hidden_dim

        # entity only have feature part but no logic part
        self.entity_feature_embedding = nn.Embedding(nentity, self.entity_dim)

        self.timestamp_time_feature_embedding = nn.Embedding(ntimestamp, self.timestamp_dim)

        self.relation_feature_embedding = nn.Embedding(nrelation, self.relation_dim)
        self.relation_logic_embedding = nn.Embedding(nrelation, self.relation_dim)
        self.relation_time_feature_embedding = nn.Embedding(nrelation, self.relation_dim)
        self.relation_time_logic_embedding = nn.Embedding(nrelation, self.relation_dim)

        self.entity_projection = EntityProjection(hidden_dim, drop=drop)
        self.entity_intersection = EntityIntersection(hidden_dim)
        self.entity_union = EntityUnion(hidden_dim)
        self.entity_negation = EntityNegation(hidden_dim)

        self.time_projection = TimeProjection(hidden_dim, drop=drop)
        self.time_intersection = TemporalIntersection(hidden_dim)
        self.time_union = TemporalUnion(hidden_dim)
        self.time_negation = TemporalNegation(hidden_dim)
        self.time_before = TemporalBefore(hidden_dim)
        self.time_after = TemporalAfter(hidden_dim)
        self.time_next = TemporalNext()

        self.batch_entity_range = torch.arange(nentity).float().repeat(test_batch_size, 1)
        self.epsilon = 2.0
        self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=False)
        self.embedding_range = nn.Parameter(torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), requires_grad=False)
        embedding_range = self.embedding_range.item()
        self.modulus = nn.Parameter(torch.Tensor([0.5 * embedding_range]), requires_grad=True)
        self.cen = center_reg
        self.parser = self.build_parser()
    
    
    def build_parser(self):
        def And(q1, q2):
            q1_feature, q1_logic, q1_time_feature, q1_time_logic = q1
            q2_feature, q2_logic, q2_time_feature, q2_time_logic = q2
            feature = torch.stack([q1_feature, q2_feature])
            logic = torch.stack([q1_logic, q2_logic])
            time_feature = torch.stack([q1_time_feature, q2_time_feature])
            time_logic = torch.stack([q1_time_logic, q2_time_logic])
            return self.entity_intersection(feature, logic, time_feature, time_logic)

        def And3(q1, q2, q3):
            q1_feature, q1_logic, q1_time_feature, q1_time_logic = q1
            q2_feature, q2_logic, q2_time_feature, q2_time_logic = q2
            q3_feature, q3_logic, q3_time_feature, q3_time_logic = q3
            feature = torch.stack([q1_feature, q2_feature, q3_feature])
            logic = torch.stack([q1_logic, q2_logic, q3_logic])
            time_feature = torch.stack([q1_time_feature, q2_time_feature, q3_time_feature])
            time_logic = torch.stack([q1_time_logic, q2_time_logic, q3_time_logic])
            return self.entity_intersection(feature, logic, time_feature, time_logic)

        def Or(q1, q2):
            q1_feature, q1_logic, q1_time_feature, q1_time_logic = q1
            q2_feature, q2_logic, q2_time_feature, q2_time_logic = q2
            feature = torch.stack([q1_feature, q2_feature])
            logic = torch.stack([q1_logic, q2_logic])
            time_feature = torch.stack([q1_time_feature, q2_time_feature])
            time_logic = torch.stack([q1_time_logic, q2_time_logic])
            return self.entity_union(feature, logic, time_feature, time_logic)

        def Not(q):
            feature, logic, time_feature, time_logic = q
            return self.entity_negation(feature, logic, time_feature, time_logic)

        def TimeNot(q):
            feature, logic, time_feature, time_logic = q
            return self.time_negation(feature, logic, time_feature, time_logic)

        def EntityProjection2(e1, r1, t1):
            s_feature, s_logic, s_time_feature, s_time_logic = e1
            r_feature, r_logic, r_time_feature, r_time_logic = r1
            t_feature, t_logic, t_time_feature, t_time_logic = t1
            return self.entity_projection(
                s_feature, s_logic, s_time_feature, s_time_logic,
                r_feature, r_logic, r_time_feature, r_time_logic,
                t_feature, t_logic, t_time_feature, t_time_logic
            )

        def TimeProjection2(e1, r1, e2):
            s_feature, s_logic, s_time_feature, s_time_logic = e1
            r_feature, r_logic, r_time_feature, r_time_logic = r1
            o_feature, o_logic, o_time_feature, o_time_logic = e2
            return self.time_projection(
                s_feature, s_logic, s_time_feature, s_time_logic,
                r_feature, r_logic, r_time_feature, r_time_logic,
                o_feature, o_logic, o_time_feature, o_time_logic
            )

        def TimeAnd(q1, q2):
            q1_feature, q1_logic, q1_time_feature, q1_time_logic = q1
            q2_feature, q2_logic, q2_time_feature, q2_time_logic = q2
            feature = torch.stack([q1_feature, q2_feature])
            logic = torch.stack([q1_logic, q2_logic])
            time_feature = torch.stack([q1_time_feature, q2_time_feature])
            time_logic = torch.stack([q1_time_logic, q2_time_logic])
            return self.time_intersection(feature, logic, time_feature, time_logic)

        def TimeAnd3(q1, q2, q3):
            q1_feature, q1_logic, q1_time_feature, q1_time_logic = q1
            q2_feature, q2_logic, q2_time_feature, q2_time_logic = q2
            q3_feature, q3_logic, q3_time_feature, q3_time_logic = q3
            feature = torch.stack([q1_feature, q2_feature, q3_feature])
            logic = torch.stack([q1_logic, q2_logic, q3_logic])
            time_feature = torch.stack([q1_time_feature, q2_time_feature, q3_time_feature])
            time_logic = torch.stack([q1_time_logic, q2_time_logic, q3_time_logic])
            return self.time_intersection(feature, logic, time_feature, time_logic)

        def TimeOr(q1, q2):
            q1_feature, q1_logic, q1_time_feature, q1_time_logic = q1
            q2_feature, q2_logic, q2_time_feature, q2_time_logic = q2
            feature = torch.stack([q1_feature, q2_feature])
            logic = torch.stack([q1_logic, q2_logic])
            time_feature = torch.stack([q1_time_feature, q2_time_feature])
            time_logic = torch.stack([q1_time_logic, q2_time_logic])
            return self.time_union(feature, logic, time_feature, time_logic)

        def TimeBefore(q):
            feature, logic, time_feature, time_logic = q
            return self.time_before(feature, logic, time_feature, time_logic)

        def TimeAfter(q):
            feature, logic, time_feature, time_logic = q
            return self.time_after(feature, logic, time_feature, time_logic)

        def TimeNext(q):
            feature, logic, time_feature, time_logic = q
            return self.time_next(feature, logic, time_feature, time_logic)

        def beforePt(e1, r1, e2):
            return TimeBefore(TimeProjection2(e1, r1, e2))

        def afterPt(e1, r1, e2):
            return TimeAfter(TimeProjection2(e1, r1, e2))

        neural_ops = {
            "And": And,
            "And3": And3,
            "Or": Or,
            "Not": Not,
            "EntityProjection": EntityProjection2,
            "TimeProjection": TimeProjection2,
            "TimeAnd": TimeAnd,
            "TimeAnd3": TimeAnd3,
            "TimeOr": TimeOr,
            "TimeNot": TimeNot,
            "TimeBefore": TimeBefore,
            "TimeAfter": TimeAfter,
            "TimeNext": TimeNext,
            "afterPt": afterPt,
            "beforePt": beforePt,
        }
        return Parser.NeuralParser(neural_ops)

    def entity_token(self, idx):
        feature = self.entity_feature(idx)
        logic = torch.zeros_like(feature).to(feature.device)
        time_feature = torch.zeros_like(feature).to(feature.device)
        time_logic = torch.zeros_like(feature).to(feature.device)
        return feature, logic, time_feature, time_logic

    def relation_token(self, idx):
        feature = convert_to_feature(self.scale(self.relation_feature_embedding(idx)))
        logic = convert_to_logic(self.scale(self.relation_logic_embedding(idx)))
        time_feature = convert_to_time_feature(self.scale(self.relation_time_feature_embedding(idx)))
        time_logic = convert_to_time_logic(self.scale(self.relation_time_logic_embedding(idx)))
        return feature, logic, time_feature, time_logic

    def timestamp_token(self, idx):
        time_feature = self.timestamp_feature(idx)
        feature = torch.zeros_like(time_feature).to(time_feature.device)
        logic = torch.zeros_like(feature).to(feature.device)
        time_logic = torch.zeros_like(feature).to(feature.device)
        return feature, logic, time_feature, time_logic

    def scale(self, embedding):
        return embedding / self.embedding_range
        
    def fast_function(self, func_name):
        if func_name in self.func_cache:
            return self.func_cache[func_name]
        func = self.eval(func_name)
        self.func_cache[func_name] = func
        return func
        
    def embed_args(self, query_args, query_tensor: torch.Tensor):
        embedding_of_args = []
        for i in range(len(query_args)):
            arg_name = query_args[i]
            tensor = query_tensor[:, i]
            if is_entity(arg_name):
                token_embedding = self.entity_token(tensor)
            elif is_relation(arg_name):
                token_embedding = self.relation_token(tensor)
            elif is_timestamp(arg_name):
                token_embedding = self.timestamp_token(tensor)
            else:
                raise Exception("Unknown Args %s" % arg_name)
            embedding_of_args.append(token_embedding)
        return tuple(embedding_of_args)

    def distance_between_timestamp_and_query(self, timestamp_feature, time_feature, time_logic):
        d_center = timestamp_feature - time_feature
        d_left = timestamp_feature - (time_feature - time_logic)
        d_right = timestamp_feature - (time_feature + time_logic)

        # inner distance
        feature_distance = torch.abs(d_center)
        inner_distance = torch.min(feature_distance, time_logic)
        # outer distance
        outer_distance = torch.min(torch.abs(d_left), torch.abs(d_right))
        outer_distance[feature_distance < time_logic] = 0.  # if entity is inside, we don't care about outer.

        distance = torch.norm(outer_distance, p=1, dim=-1) + self.cen * torch.norm(inner_distance, p=1, dim=-1)
        return distance

    def scoring_timestamp(self, timestamp_feature, q):
        feature, logic, time_feature, time_logic = q
        distance = self.distance_between_timestamp_and_query(timestamp_feature, time_feature, time_logic)
        score = self.gamma - distance * self.modulus
        return score

    def distance_between_entity_and_query(self, entity_feature, query_feature, query_logic):
        d_center = entity_feature - query_feature
        d_left = entity_feature - (query_feature - query_logic)
        d_right = entity_feature - (query_feature + query_logic)

        # inner distance
        feature_distance = torch.abs(d_center)
        inner_distance = torch.min(feature_distance, query_logic)
        # outer distance
        outer_distance = torch.min(torch.abs(d_left), torch.abs(d_right))
        outer_distance[feature_distance < query_logic] = 0.  # if entity is inside, we don't care about outer.

        distance = torch.norm(outer_distance, p=1, dim=-1) + self.cen * torch.norm(inner_distance, p=1, dim=-1)
        return distance

    def scoring_entity(self, entity_feature, q):
        feature, logic, time_feature, time_logic = q
        distance = self.distance_between_entity_and_query(entity_feature, feature, logic)
        score = self.gamma - distance * self.modulus
        return score

    def timestamp_feature(self, idx):
        return convert_to_time_feature(self.scale(self.timestamp_time_feature_embedding(idx)))

    def entity_feature(self, idx):
        return convert_to_feature(self.scale(self.entity_feature_embedding(idx)))

    def scoring_to_answers(self, answer_ids: torch.Tensor, q, predict_entity=True, DNF_predict=False):
        q = tuple([i.unsqueeze(dim=2) for i in q])
        if predict_entity:
            feature = self.entity_feature(answer_ids).unsqueeze(dim=1)
            scores = self.scoring_entity(feature, q)
        else:
            feature = self.timestamp_feature(answer_ids).unsqueeze(dim=1)
            scores = self.scoring_timestamp(feature, q)

        if DNF_predict:
            scores = torch.max(scores, dim=1)[0]
        else:
            scores = scores.squeeze(dim=1)
        return scores

    def forward_predict(self, query_structure, query_tensor: torch.Tensor, answer: torch.Tensor) -> torch.Tensor:
        # This version of forward_predict assumes query_structure to be 'Pe', in other words it only handles entity prediction.
        query_name = query_structure
        query_args = self.parser.fast_args(query_name)

        func = self.parser.fast_function(query_name)
        embedding_of_args = self.embed_args(query_args, query_tensor)
        predict = func(*embedding_of_args)
        all_predict = tuple([i.unsqueeze(dim=1) for i in predict])
        if is_to_predict_entity_set(query_name):
            return self.scoring_to_answers(answer, all_predict, predict_entity=True, DNF_predict=False)
        else:
            return self.scoring_to_answers(answer, all_predict, predict_entity=False, DNF_predict=False)