
import os
import json

from pathlib import Path

from statistics.measure import Measure

overall_scores = {
    "DE_TransE": {
        "HIT1": 0.107,
        "HIT3": 0.471,
        "HIT10": 0.693,
        "MR": 1.0,
        "MRR": 0.321
    },
    "DE_SimplE": {
        "HIT1": 0.412,
        "HIT3": 0.591,
        "HIT10": 0.732,
        "MR": 1.0,
        "MRR": 0.523
    },
    "DE_DistMult": {
        "HIT1": 0.391,
        "HIT3": 0.565,
        "HIT10": 0.705,
        "MR": 1.0,
        "MRR": 0.500
    },
    "TERO": {
        "HIT1": 0.467,
        "HIT3": 0.616,
        "HIT10": 0.731,
        "MR": 1.0,
        "MRR": 0.599
    },
    "ATISE": {
        "HIT1": 0.448,
        "HIT3": 0.644,
        "HIT10": 0.757,
        "MR": 1.0,
        "MRR": 0.561
    },
    "TFLEX": {
        "HIT1": 0.314,
        "HIT3": 0.491,
        "HIT10": 0.636,
        "MR": 1.0,
        "MRR": 0.426
    }
}


class Statistics():
    def __init__(self, params) -> None:
        self.params = params

    def hypothesis_1(self, ranked_quads, embeddings, normalization_scores = None):
        for element_type in ["HEAD", "RELATION", "TAIL", "TIME"]:
            print("Rank of question tuples when " + str(element_type) + " is the answer element:")
            
            measure = Measure()

            for quad in ranked_quads:
                if quad[element_type] is not "0":
                    continue

                ranks = {}
                for embedding in embeddings:
                    if embedding == "TFLEX":
                        if element_type not in ["TAIL", "TIME"]:
                            continue
                    
                    ranks[embedding] = int(float(quad["RANK"][embedding]))
                measure.update(ranks)
            
            measure.normalize()
            if normalization_scores is not None:
                for embedding in embeddings:
                    measure.hit1 /= normalization_scores[embedding]["HIT1"]
                    measure.hit3 /= normalization_scores[embedding]["HIT3"]
                    measure.hit10 /= normalization_scores[embedding]["HIT10"]
                    measure.mr /= normalization_scores[embedding]["MR"]
                    measure.mrr /= normalization_scores[embedding]["MRR"]

            measure.print()

            results_path = os.path.join(self.params.base_directory, "result", self.params.dataset, "hypothesis_1", str(element_type).lower()+".json")

            Path(results_path).touch(exist_ok=True)
            out_file = open(results_path, "w", encoding="utf8")
            json.dump(measure.as_dict(), out_file, indent=4)
            out_file.close()

    def hypothesis_2(self, ranked_quads, embeddings):
        for element in ["ENTITY", "RELATION", "TIME"]:
            print("Testing hypothesis 2 on " + str(element) + "s:")
            element_measures = {}
            json_output = []

            if element is "ENTITY":
                target_parts = ["HEAD", "TAIL"]
            else:
                target_parts = [element]

            for target_part in target_parts:
                for quad in ranked_quads:
                    if quad[target_part] is "0":
                        continue

                    if quad[target_part] not in element_measures.keys():
                        element_measures[quad[target_part]] = Measure()
                    
                    ranks = {}
                    for embedding in embeddings:
                        ranks[embedding] = int(float(quad["RANK"][embedding]))
                    element_measures[quad[target_part]].update(ranks)
            
            for element_key in element_measures.keys():
                element_measures[element_key].normalize()

                json_output.append({element: element_key, "NUM_FACTS": element_measures[element_key].num_facts, "MEASURE": element_measures[element_key].as_dict()})

                print(str(element) + ": "+str(element_key) + ":")
                element_measures[element_key].print()

            json_output.sort(key=lambda val: val["NUM_FACTS"], reverse=True)

            results_path = os.path.join(self.params.base_directory, "result", self.params.dataset, "hypothesis_2_"+str(element).lower()+".json")
            Path(results_path).touch(exist_ok=True)
            out_file = open(results_path, "w", encoding="utf8")
            json.dump(json_output, out_file, indent=4)
            out_file.close()

    def hypothesis_3(self, ranked_quads, embeddings):
        for element_type in ["HEAD", "TAIL"]:
            json_output = []

            print("Testing hypothesis 3 on " + str(element_type) + " relations:")

            entity_measures = {}
            for quad in ranked_quads:
                if quad[element_type] is "0":
                    continue

                if quad[element_type] not in entity_measures.keys():
                    entity_measures[quad[element_type]] = Measure()
                
                ranks = {}
                for embedding in embeddings:
                    ranks[embedding] = int(float(quad["RANK"][embedding]))
                entity_measures[quad[element_type]].update(ranks)
            
            for entity in entity_measures.keys():
                entity_measures[entity].normalize()

                min_val = 1
                max_val = 0
                for embedding in embeddings:
                    if entity_measures[entity].mrr[embedding] < min_val:
                        min_val = entity_measures[entity].mrr[embedding]
                    if entity_measures[entity].mrr[embedding] > max_val:
                        max_val = entity_measures[entity].mrr[embedding]

                json_output.append({"ENTITY": entity, "NUM_FACTS": entity_measures[entity].num_facts, "DIFFERENCE": {"MRR": max_val-min_val}, "MEASURE": entity_measures[entity].as_dict()})
                print("Entity: "+str(entity)+": (Difference: " + str(max_val-min_val) + ")")
                entity_measures[entity].print()
            
            json_output.sort(key=lambda val: val["NUM_FACTS"], reverse=True)

            results_path = os.path.join(self.params.base_directory, "result", self.params.dataset, "hypothesis_3_"+str(element_type).lower()+".json")

            Path(results_path).touch(exist_ok=True)
            out_file = open(results_path, "w", encoding="utf8")
            json.dump(json_output, out_file, indent=4)
            out_file.close()

    def run(self):
        ranks_path = os.path.join(self.params.base_directory, "result", self.params.dataset, "ranked_quads.json")
        
        in_file = open(ranks_path, "r", encoding="utf8")
        ranked_quads = json.load(in_file)
        in_file.close()
        
        embeddings = ["DE_TransE", "DE_SimplE", "DE_DistMult", "TERO", "ATISE", "TFLEX"]

        self.hypothesis_1(ranked_quads, embeddings, overall_scores)
        #self.hypothesis_2(ranked_quads, embeddings)
        #self.hypothesis_3(ranked_quads, embeddings)
