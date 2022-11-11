
import torch
import sys
from de_simple import de_transe, de_simple, de_distmult, dataset, params

class Loader:
    def __init__(self, params, model_path, embedding):
        self.params = params
        self.model_path = model_path
        self.embedding = embedding

    def load(self):
        old_modules = sys.modules

        if self.embedding in ["DE_TransE", "DE_SimplE", "DE_DistMult"]:
            sys.modules['de_transe'] = de_transe
            sys.modules['de_simple'] = de_simple
            sys.modules['de_distmult'] = de_distmult
            sys.modules['dataset'] = dataset
            sys.modules['params'] = params

        model = torch.load(self.model_path, map_location="cpu")
        sys.modules = old_modules
        return model
