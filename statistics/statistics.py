
import os
import json

from pathlib import Path
from statistics.measure import Measure


class Statistics():
    def __init__(self, params) -> None:
        self.params = params

    def write_json(self, path, dict):
        Path(path).touch(exist_ok=True)
        out_file = open(path, "w", encoding="utf8")
        json.dump(dict, out_file, indent=4)
        out_file.close()
    
    def read_json(self, path):
        in_file = open(path, "r", encoding="utf8")
        dict = json.load(in_file)
        in_file.close()
        return dict
    
    def calculate_overall_scores(self, ranked_quads, embeddings):
        print("Rank of all question tuples:")
        
        measure = Measure()

        for quad in ranked_quads:
            # if not (quad["TAIL"] == "0" or quad["HEAD"] == "0"):
            #     continue

            ranks = {}
            for embedding in embeddings:
                if embedding == "TFLEX":
                    if not (quad["TAIL"] == "0" or quad["TIME"] == "0"):
                        continue
                
                ranks[embedding] = int(float(quad["RANK"][embedding]))
            measure.update(ranks)
        
        measure.normalize()
        measure.print()

        results_path = os.path.join(self.params.base_directory, "result", self.params.dataset, "overall_scores.json")
        self.write_json(results_path, measure.as_dict())

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
            
            measure.print()
            measure.normalize()

            results_path = os.path.join(self.params.base_directory, "result", self.params.dataset, "hypothesis_1", str(element_type).lower()+".json")
            self.write_json(results_path, measure.as_dict())

            if normalization_scores is not None:
                measure.normalize_to(normalization_scores)
                results_path = os.path.join(self.params.base_directory, "result", self.params.dataset, "hypothesis_1", str(element_type).lower()+"_normalized.json")
                self.write_json(results_path, measure.as_dict())

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
        embeddings = ["DE_TransE", "DE_SimplE", "DE_DistMult", "TERO", "ATISE", "TFLEX"]

        ranks_path = os.path.join(self.params.base_directory, "result", self.params.dataset, "ranked_quads.json")
        ranked_quads = self.read_json(ranks_path)

        self.calculate_overall_scores(ranked_quads, embeddings)

        overall_scores_path = os.path.join(self.params.base_directory, "result", self.params.dataset, "overall_scores.json")        
        overall_scores = self.read_json(overall_scores_path)

        self.hypothesis_1(ranked_quads, embeddings, overall_scores)
        #self.hypothesis_2(ranked_quads, embeddings)
        #self.hypothesis_3(ranked_quads, embeddings)
