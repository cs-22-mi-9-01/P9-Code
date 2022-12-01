class Measure():
    def __init__(self):
        self.hit1 = {}
        self.hit3 = {}
        self.hit10 = {}
        self.mrr = {}
        self.mr = {}
        self.num_facts = {}
    
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
        if embedding not in self.num_facts.keys():
            self.num_facts = 0

    def update(self, ranks):
        for embedding in ranks.keys():
            self.initialize_embedding(embedding)

            self.num_facts[embedding] += 1

            if ranks[embedding] == 1:
                self.hit1[embedding] += 1.0
            if ranks[embedding] <= 3:
                self.hit3[embedding] += 1.0
            if ranks[embedding] <= 10:
                self.hit10[embedding] += 1.0

            self.mr[embedding] += ranks[embedding]
            self.mrr[embedding] += (1.0 / ranks[embedding])

    def normalize(self):
        for embedding in self.hit1:
            self.hit1[embedding] /= self.num_facts[embedding]
            self.hit3[embedding] /= self.num_facts[embedding]
            self.hit10[embedding] /= self.num_facts[embedding]
            self.mr[embedding] /= self.num_facts[embedding]
            self.mrr[embedding] /= self.num_facts[embedding]
    
    def normalize_to(self, scores):
        for embedding in self.hit1:            
            self.hit1[embedding] /= scores[embedding]["HIT1"]
            self.hit3[embedding] /= scores[embedding]["HIT3"]
            self.hit10[embedding] /= scores[embedding]["HIT10"]
            #self.mr[embedding] /= scores[embedding]["MR"]
            self.mrr[embedding] /= scores[embedding]["MRR"]

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

    def as_dict(self):
        ret_dict = {}
        for embedding in self.hit1.keys():
            ret_dict[embedding] = {
                "HIT1": self.hit1[embedding], 
                "HIT3": self.hit3[embedding], 
                "HIT10": self.hit10[embedding], 
                "MR": self.mr[embedding], 
                "MRR": self.mrr[embedding]
            }
        return ret_dict