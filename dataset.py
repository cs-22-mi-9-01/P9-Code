
import os
import torch

from torch.utils.data import Dataset
from datetime import date
from torch.utils.data.dataset import T_co


class KnowledgeGraphDataset(Dataset):
    def __init__(self, params):
        super(KnowledgeGraphDataset, self).__init__()
        self.params = params
        self.name = params.dataset
        self.path = os.path.join(params.base_directory, "data", self.name.lower())

        self.next_id = {"entity": 0, "relation": 0}
        self.ids = {"entity": {}, "relation": {}}
        self.entity_set = set()
        self.relation_set = set()
        self.data = {"train": self.read_file(os.path.join(self.path, "train.txt")),
                     "valid": self.read_file(os.path.join(self.path, "valid.txt")),
                     "test": self.read_file(os.path.join(self.path, "test.txt"))}
        self.target_dataset = "train"

        for split in ["train", "valid", "test"]:
            self.data[split] = torch.tensor(self.data[split]).long().to(self.params.device)

    def __len__(self):
        return len(self.data[self.target_dataset])

    def __getitem__(self, index) -> T_co:
        return self.data[self.target_dataset][index]

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

            facts.append([head, rel, tail, year, month, day])
        return facts

    def to_id(self, type, element):
        if element not in self.ids[type].keys():
            self.ids[type][element] = self.next_id[type]
            self.next_id[type] = self.next_id[type] + 1
        return self.ids[type][element]

    def split_timestamp(self, element):
        dt = date.fromisoformat(element)

        return dt.year, dt.month, dt.day

    def num_of_entities(self):
        return len(self.ids["entity"])

    def num_of_relations(self):
        return len(self.ids["relation"])

    def get_all_facts(self):
        return self.data[self.target_dataset]
