
import os

import torch
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
        self.data_ids = {"train": self.read_file(os.path.join(self.path, "train.txt")),
                        "valid": self.read_file(os.path.join(self.path, "valid.txt")),
                        "test": self.read_file(os.path.join(self.path, "test.txt"))}

    def __len__(self):
        return len(self.data_ids["train"])

    def __getitem__(self, index) -> T_co:
        return self.data_ids["train"][index]

    def read_file(self, filename):
        with open(filename, "r", encoding="utf8") as f:
            data = f.readlines()

        facts = []
        for line in data:
            elements = line.strip().split("\t")

            head = self.to_id("entity", elements[0])
            rel = self.to_id("relation", elements[1])
            tail = self.to_id("entity", elements[2])
            timestamp = self.to_id("time", elements[3])

            facts.append([head, rel, tail, timestamp])
        return facts

    def to_id(self, type, element):
        if element not in self.ids[type].keys():
            self.ids[type][element] = self.next_id[type]
            self.next_id[type] = self.next_id[type] + 1
        return self.ids[type][element]

    def to_timestamp(self, element):
        return date.fromisoformat(element)
