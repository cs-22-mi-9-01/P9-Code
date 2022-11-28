
import torch
import sys
from de_simple import de_transe, de_simple, de_distmult, dataset, params
from TERO import TERO_model, Dataset

class Loader:
    def __init__(self, params, model_path, embedding):
        self.params = params
        self.model_path = model_path
        self.embedding = embedding

    def remove_unwanted_symbols(self, dict): 
        while True:
            target_key = None
            for key in dict.keys():
                if ' ' in key:
                    target_key = key
                    break
            if target_key is not None:
                dict[target_key.replace(' ', ' ')] = dict.pop(target_key)
            else:
                break

    def load(self):
        old_modules = sys.modules

        if self.embedding in ["DE_TransE", "DE_SimplE", "DE_DistMult"]:
            sys.modules['de_transe'] = de_transe
            sys.modules['de_simple'] = de_simple
            sys.modules['de_distmult'] = de_distmult
            sys.modules['dataset'] = dataset
            sys.modules['params'] = params
        elif self.embedding in ["TERO", "ATISE"]:
            sys.modules['model'] = TERO_model
            sys.modules['Dataset'] = Dataset

        model = torch.load(self.model_path, map_location="cpu")
        sys.modules = old_modules

        if self.embedding in ["DE_TransE", "DE_SimplE", "DE_DistMult"]:
            self.remove_unwanted_symbols(model.module.dataset.ent2id)
            self.remove_unwanted_symbols(model.module.dataset.rel2id)
        if self.embedding in ['TERO', 'ATISE']:
            self.remove_unwanted_symbols(model.kg.entity_dict)
            self.remove_unwanted_symbols(model.kg.relation_dict)
            model.gpu = False

        return model
