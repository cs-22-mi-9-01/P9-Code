import torch
import numpy as np
import time
import datetime
import json
import os

class RankCalculator:
    def __init__(self, params, model):
        self.params = params
        self.model = model

        cache_imported_path = os.path.join(params.base_directory, "TFLEX", "cache_imported")

        self.entity2id = self.read_json(os.path.join(cache_imported_path, "entity2idx.json"))
        self.relation2id = self.read_json(os.path.join(cache_imported_path, "relation2idx.json"))
        self.timestamp2id = self.read_json(os.path.join(cache_imported_path, "timestamp2idx.json"))
    
    def read_json(self, path):
        in_file = open(path, "r", encoding="utf8")
        data = json.load(in_file)
        in_file.close()
        return data

    def get_rank(self, scores):  # assuming the first fact is the correct fact
        return torch.sum((scores < scores[0]).float()).item() + 1

    def simulate_query(self, head, relation, tail, time, target):
        query = None

        if target == "t":
            query = [self.entity2id[head], self.relation2id[relation], self.timestamp2id[time]]

        return torch.tensor([query])
    
    def simulate_answers(self, target, answer):
        answers = None

        if target == "t":
            answers = [i for i in range(0, self.model.nentity)]
            answers = [self.entity2id[answer]] + answers
        
        return torch.tensor([answers])


    def get_rank_of(self, head, relation, tail, time, answer):
        target = "?"
        if head == "0":
            target = "h"
        elif relation == "0":
            target = "r"
        elif tail == "0":
            target = "t"
        elif time == "0":
            target = "T"

        query = self.simulate_query(head, relation, tail, time, target)
        candidate_answers = self.simulate_answers(target, answer)
        sim_scores = self.model.forward_predict('Pe', query, candidate_answers)
        rank = self.get_rank(sim_scores[0])

        return rank
