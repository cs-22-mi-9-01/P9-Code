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
        head, rel, tail, year, month, day = fact
        if head_or_tail == "head":
            generated_facts = torch.tensor([[i, rel, tail, year, month, day] for i in range(self.params.dataset.num_of_entities()) if i != head])
        if head_or_tail == "tail":
            generated_facts = torch.tensor([[head, rel, i, year, month, day] for i in range(self.params.dataset.num_of_entities()) if i != tail])

        facts = torch.cat((fact.view(1, 6), generated_facts), dim=0)
        facts.to(self.params.device)
        return facts

    def get_rank(self, sim_scores): # Assuming the first fact is the true fact
        aggregate = [(1 if s <= sim_scores[0] else 0) for s in sim_scores]
        return sum(aggregate)

    def test(self):
        print("--------- Starting test ---------")

        num_of_facts = self.params.dataset.num_of_facts()

        for i, fact in enumerate(self.params.dataset.get_all_facts()):
            print("Evaluating fact " + str(i) + "/" + str(num_of_facts) + " ...")
            for head_or_tail in ["head", "tail"]:
                simulated_facts = self.expand_and_replace(fact, head_or_tail)
                heads, rels, tails, years, months, days = split_facts(simulated_facts)
                sim_scores = self.model(heads, rels, tails, years, months, days)
                rank = self.get_rank(sim_scores)
                self.measure.update(rank)

        self.measure.normalize(num_of_facts)

        print("--------- Results ---------")
        self.measure.print()
