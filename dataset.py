
import os

import numpy as np
from torch.utils.data import Dataset
from datetime import date
from torch.utils.data.dataset import T_co

embedding_vec_sizes = 3

class KnowledgeGraphDataset(Dataset):
    def __init__(self, params):
        super(KnowledgeGraphDataset, self).__init__()
        self.name = params.dataset
        self.path = os.path.join(params.base_directory, "data", self.name.lower())
        self.next_id = {"entity": 0, "relation": 0, "time": 0}
        self.ids = {"entity": {}, "relation": {}, "time": {}}
        self.entity_set = set()
        self.relation_set = set()
        self.data = {"train": self.read_file(os.path.join(self.path, "train.txt")),
                        "valid": self.read_file(os.path.join(self.path, "valid.txt")),
                        "test": self.read_file(os.path.join(self.path, "test.txt"))}

        for split in ["train", "valid", "test"]:
            self.data[split] = np.array(self.data[split])

    def __len__(self):
        return len(self.data["train"])

    def __getitem__(self, index) -> T_co:
        return self.data["train"][index]

    def read_file(self, filename):
        with open(filename, "r", encoding="utf8") as f:
            data = f.readlines()

        facts = []
        for line in data:
            elements = line.strip().split("\t")

            head = self.to_id("entity", elements[0])
            rel = self.to_id("relation", elements[1])
            tail = self.to_id("entity", elements[2])
            year, month, day = self.split_timestamp(elements[3])

            self.register_entity(head)
            self.register_relation(rel)
            self.register_entity(tail)
            facts.append([head, rel, tail, year, month, day])
        return facts

    def split_timestamp(self, element):
        dt = date.fromisoformat(element)

        return dt.year, dt.month, dt.day

    def register_entity(self, entity):
        self.entity_set.add(entity)

    def register_relation(self, relation):
        self.relation_set.add(relation)

    def num_of_entities(self):
        return len(self.entity_set)

    def num_of_relations(self):
        return len(self.relation_set)

    def to_id(self, type, element):
        if element not in self.ids[type].keys():
            self.ids[type][element] = self.next_id[type]
            self.next_id[type] = self.next_id[type] + 1
        return self.ids[type][element]

    def get_all_data(self):
        return ([c[0] for c in self.data["train"]],
                [c[1] for c in self.data["train"]],
                [c[2] for c in self.data["train"]],
                [c[3] for c in self.data["train"]],
                [c[4] for c in self.data["train"]],
                [c[5] for c in self.data["train"]])
