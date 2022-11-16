class Measure():
    def __init__(self):
        self.hit1 = {}
        self.hit3 = {}
        self.hit10 = {}
        self.mrr = {}
        self.mr = {}
        self.num_facts = 0
    
    def initialize_embedding(self, embedding):
        if embedding not in self.hit1.keys():
            self.hit1[embedding] = 0
        if embedding not in self.hit3.keys():
            self.hit3[embedding] = 0
        if embedding not in self.hit10.keys():
            self.hit10[embedding] = 0
        if embedding not in self.mrr.keys():
            self.mrr[embedding] = 0
        if embedding not in self.mr.keys():
            self.mr[embedding] = 0

    def update(self, embedding, rank):
        self.initialize_embedding(embedding)
        self.num_facts += 1

        if rank == 1:
            self.hit1[embedding] += 1.0
        if rank <= 3:
            self.hit3[embedding] += 1.0
        if rank <= 10:
            self.hit10[embedding] += 1.0

        self.mr[embedding] += rank
        self.mrr[embedding] += (1.0 / rank)

    def normalize(self):
        for embedding in self.hit1:
            self.hit1[embedding] /= self.num_facts
            self.hit3[embedding] /= self.num_facts
            self.hit10[embedding] /= self.num_facts
            self.mr[embedding] /= self.num_facts
            self.mrr[embedding] /= self.num_facts

    def print(self):
        for embedding in self.hit1:
            print(str(embedding) + ":")
            print("Number of facts: " + str(self.num_facts))
            print("\tHit@1 =", self.hit1[embedding])
            print("\tHit@3 =", self.hit3[embedding])
            print("\tHit@10 =", self.hit10[embedding])
            print("\tMR =", self.mr[embedding])
            print("\tMRR =", self.mrr[embedding])
            print("")