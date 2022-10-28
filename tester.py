import torch
from scripts import split_facts
from measure import Measure


class Tester:
    def __init__(self, params):
        self.params = params
        self.model = torch.load(self.params.model_path)
        self.params.model = self.model
        self.model.eval()
        self.valid_or_test = "test"
        self.params.dataset.target_dataset = self.valid_or_test
        self.measure = Measure()

    def expand_and_replace(self, fact, head_or_tail):
        head, rel, tail, years, months, days = fact
        if head_or_tail == "head":
            facts = torch.tensor([[i, rel, tail, years, months, days] for i in range(self.params.dataset.num_of_entities())])
        if head_or_tail == "tail":
            facts = torch.tensor([[head, rel, i, years, months, days] for i in range(self.params.dataset.num_of_entities())])

        facts.to(self.params.device)
        return facts

    def test(self):
        for fact in self.params.dataset.get_all_facts():
            for head_or_tail in ["head", "tail"]:
                heads, rels, tails, years, months, days = split_facts(self.expand_and_replace(fact, head_or_tail))
                sim_scores = self.model(heads, rels, tails, years, months, days)
                self.measure.update(rank, raw_or_fil)

