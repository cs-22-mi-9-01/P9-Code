
from de_simple.rank_calculator import RankCalculator as DE_Rank
from TERO.rank_calculator import RankCalculator as TERO_Rank
from TFLEX.rank_calculator import RankCalculator as TFLEX_Rank

class Ranker:
    def __init__(self, params, quads, model, embedding_name):
        self.params = params
        self.quads = quads
        self.model = model
        self.embedding_name = embedding_name

    def rank(self):
        ranked_quads = []
        
        if self.embedding_name in ["DE_TransE", "DE_SimplE", "DE_DistMult"]:
            rank_calculator = DE_Rank(self.params, self.model)
        if self.embedding_name in ["TERO"]:
            rank_calculator = TERO_Rank(self.params, self.model)
        if self.embedding_name in ["TFLEX"]:
            rank_calculator = TFLEX_Rank(self.params, self.model)

        for i, quad in zip(range(0, len(self.quads)), self.quads):
            if i % 100 == 0:
                print("Ranking fact " + str(i) + "-" + str(i + 99) + " (total number: " + str(len(self.quads)) + ") with embedding " + self.embedding_name)

            if self.embedding_name in ["TFLEX"]:
                if not (quad["TAIL"] == "0" or quad["TIME"] == "0"):
                    ranked_quads.append(quad)
                    continue

            ranked_quad = quad
            if "RANK" not in ranked_quad.keys():
                ranked_quad["RANK"] = {}

            ranked_quad["RANK"][self.embedding_name] = str(rank_calculator.get_rank_of(quad["HEAD"], quad["RELATION"],
                                                                                       quad["TAIL"], quad["TIME"],
                                                                                       quad["ANSWER"]))
            ranked_quads.append(ranked_quad)

        return ranked_quads

