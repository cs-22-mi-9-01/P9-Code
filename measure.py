
class Measure:
    def __init__(self):
        self.hit1 = 0.0
        self.hit3 = 0.0
        self.hit10 = 0.0
        self.mrr = 0.0
        self.mr = 0.0

    def update(self, rank):
        if rank == 1:
            self.hit1 += 1.0
        if rank <= 3:
            self.hit3 += 1.0
        if rank <= 10:
            self.hit10 += 1.0
        self.mr += rank
        self.mrr += (1.0 / rank)

    def normalize(self, num_facts):
        self.hit1 /= (2 * num_facts)
        self.hit3 /= (2 * num_facts)
        self.hit10 /= (2 * num_facts)
        self.mr /= (2 * num_facts)
        self.mrr /= (2 * num_facts)

    def print(self):
        print("\tHit@1 =", self.hit1)
        print("\tHit@3 =", self.hit3)
        print("\tHit@10 =", self.hit10)
        print("\tMR =", self.mr)
        print("\tMRR =", self.mrr)