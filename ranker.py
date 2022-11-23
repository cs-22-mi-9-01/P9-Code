
from de_simple.rank_calculator import RankCalculator as DE_Rank

from TERO.rank_calculator import RankCalculator as TERO_Rank

class Ranker:
    def __init__(self, params, quads, model, embedding_name):
        self.params = params
        self.quads = quads
        self.model = model
        self.embedding_name = embedding_name

    def rank(self):
        ranked_quads = []
        for quad in self.quads:
            ranked_quad = quad
            if "RANK" not in ranked_quad.keys():
                ranked_quad["RANK"] = {}

            if self.embedding_name in ["DE_TransE", "DE_SimplE", "DE_DistMult"]:
                rank_calculator = DE_Rank(self.params, self.model.module.dataset, self.model)
            if self.embedding_name in ["TERO"]:
                rank_calculator = DE_Rank(self.params, self.model.modules.dataset, self.model)

            ranked_quad["RANK"][self.embedding_name] = str(rank_calculator.get_rank_of(quad["HEAD"], quad["RELATION"],
                                                                                       quad["TAIL"], quad["TIME"],
                                                                                       quad["ANSWER"]))
            ranked_quads.append(ranked_quad)

        return ranked_quads

