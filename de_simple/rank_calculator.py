import numpy as np
import torch
import datetime
from datetime import date


class RankCalculator:
    def __init__(self, params, model):
        self.params = params
        self.dataset = model.module.dataset
        self.model = model

        self.num_of_ent = self.dataset.numEnt()
        self.num_of_rel = self.dataset.numRel()

    def get_rank(self, sim_scores):  # assuming the test fact is the first one
        return (sim_scores > sim_scores[0]).sum() + 1

    def split_timestamp(self, element):
        dt = date.fromisoformat(element)
        return dt.year, dt.month, dt.day

    def shred_facts(self, facts): #takes a batch of facts and shreds it into its columns
        device = torch.device("cpu")
        heads = torch.tensor(facts[:, 0]).long().to(device)
        rels = torch.tensor(facts[:, 1]).long().to(device)
        tails = torch.tensor(facts[:, 2]).long().to(device)
        years = torch.tensor(facts[:, 3]).float().to(device)
        months = torch.tensor(facts[:, 4]).float().to(device)
        days = torch.tensor(facts[:, 5]).float().to(device)
        return heads, rels, tails, years, months, days

    def get_ent_id(self, entity):
        entity_id = self.dataset.getEntID(entity)
        if entity_id >= self.num_of_ent:
            raise Exception("Fact contains an entity that is not seen in the training set (" + str(entity) + ")")
        return entity_id

    def get_rel_id(self, relation):
        rel_id = self.dataset.getRelID(relation)
        if rel_id >= self.num_of_rel:
            raise Exception("Fact contains a relation that is not seen in the training set (" + str(relation) + ")")
        return rel_id

    def simulate_facts(self, head, relation, tail, time, target, answer):
        if head != "0":
            head = self.get_ent_id(head)
        if relation != "0":
            relation = self.get_rel_id(relation)
        if tail != "0":
            tail = self.get_ent_id(tail)
        if time != "0":
            year, month, day = self.split_timestamp(time)

        match target:
            case "h":
                sim_facts = [(i, relation, tail, year, month, day) for i in range(self.num_of_ent)]
                sim_facts = [(self.get_ent_id(answer), relation, tail, year, month, day)] + sim_facts
            case "r":
                sim_facts = [(head, i, tail, year, month, day) for i in range(self.num_of_rel)]
                sim_facts = [(head, self.get_rel_id(answer), tail, year, month, day)] + sim_facts
            case "t":
                sim_facts = [(head, relation, i, year, month, day) for i in range(self.num_of_ent)]
                sim_facts = [(head, relation, self.get_ent_id(answer), year, month, day)] + sim_facts
            case "T":
                sim_facts = []
                sim_date = date(2014, 1, 1)
                while sim_date != date(2015, 1, 1):
                    year = sim_date.year
                    month = sim_date.month
                    day = sim_date.day
                    sim_facts.append((head, relation, tail, year, month, day))
                    sim_date = sim_date + datetime.timedelta(days=1)

                year, month, day = self.split_timestamp(answer)
                sim_facts = [(head, relation, tail, year, month, day)] + sim_facts
            case _:
                raise Exception("Unknown target")

        return self.shred_facts(np.array(sim_facts))

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

        heads, rels, tails, years, months, days = self.simulate_facts(head, relation, tail, time, target, answer)
        sim_scores = self.model.module(heads, rels, tails, years, months, days).cpu().data.numpy()
        rank = self.get_rank(sim_scores)

        return rank
