
import os
import json
import pandas

from pathlib import Path
from statistics.measure import Measure
from copy import deepcopy


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
    
    def read_csv(self, path):
        in_file = open(path, "r", encoding="utf8")
        csv = pandas.read_csv(in_file, delimiter='\t')
        in_file.close()
        return csv
    
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

    def hypothesis_2(self, ranked_quads, embeddings, normalization_scores = None):
        for element in ["ENTITY", "RELATION", "TIME"]:
            print("Testing hypothesis 2 on " + str(element) + "s:")
            element_measures = {}
            json_output = []
            json_output_normalized = []

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
                        if embedding == "TFLEX":
                            if not (quad["TAIL"] == "0" or quad["TIME"] == "0"):
                                continue

                        ranks[embedding] = int(float(quad["RANK"][embedding]))
                    element_measures[quad[target_part]].update(ranks)
            
            for element_key in element_measures.keys():
                element_measures[element_key].normalize()

                json_output.append({"ELEMENT": element_key, "NUM_FACTS": max(element_measures[element_key].num_facts.values()), "MEASURE": element_measures[element_key].as_dict()})
                if normalization_scores is not None:
                    element_measures[element_key].normalize_to(normalization_scores)
                    json_output_normalized.append({"ELEMENT": element_key, "NUM_FACTS": max(element_measures[element_key].num_facts.values()), "MEASURE": element_measures[element_key].as_dict()})

                #print(str(element) + ": "+str(element_key) + ":")
                #element_measures[element_key].print()

            json_output.sort(key=lambda val: val["NUM_FACTS"], reverse=True)
            if normalization_scores is not None:
                json_output_normalized.sort(key=lambda val: val["NUM_FACTS"], reverse=True)

            results_path = os.path.join(self.params.base_directory, "result", self.params.dataset, "hypothesis_2", str(element).lower()+".json")
            self.write_json(results_path, json_output)
            
            if normalization_scores is not None:
                results_path = os.path.join(self.params.base_directory, "result", self.params.dataset, "hypothesis_2", str(element).lower()+"_normalized.json")
                self.write_json(results_path, json_output_normalized)

    def calc_overlap(self, arr_x, arr_y):
        shared_vals = 0.0
        total_vals = 0.0

        for val in arr_x:
            if val in arr_y:
                shared_vals += 1.0
            total_vals += 1.0
        
        return shared_vals/total_vals

    def hypothesis_2_top_x(self, embeddings):
        for element in ["ENTITY", "RELATION", "TIME"]:
            print("Testing hypothesis 2 top X percent on " + str(element) + "s:")

            input_path = os.path.join(self.params.base_directory, "result", self.params.dataset, "hypothesis_2", str(element).lower()+".json")
            json_input = self.read_json(input_path)

            top_percentage = 0.3
            no_of_elements = len(json_input)
            element_split = int(no_of_elements * top_percentage)
            json_percentage = {}

            for embedding in embeddings:
                if embedding not in json_percentage.keys():
                    json_percentage[embedding] = {"TOP": [], "BOT": []}

                json_input.sort(key=lambda val: val["MEASURE"][embedding]["MRR"], reverse=False)
                for i in range(0, no_of_elements):
                    if i < element_split:
                        json_percentage[embedding]["BOT"].append(json_input[i]["ELEMENT"])
                    else:
                        json_percentage[embedding]["TOP"].append(json_input[i]["ELEMENT"])
            
            json_overlap = []

            for embedding_n in embeddings:
                for embedding_m in embeddings:
                    json_overlap.append({
                        "EMBEDDING_N": embedding_n,
                        "EMBEDDING_M": embedding_m,
                        "OVERLAP_TOP": self.calc_overlap(json_percentage[embedding_n]["TOP"], json_percentage[embedding_m]["TOP"])
                    })

            results_path = os.path.join(self.params.base_directory, "result", self.params.dataset, "hypothesis_2", 
                                        "top_x_overlap", str(element).lower()+"_top_"+str(int(top_percentage*100))+"_percentage.json")
            self.write_json(results_path, json_percentage)
            
            results_path = os.path.join(self.params.base_directory, "result", self.params.dataset, "hypothesis_2", 
                                        "top_x_overlap", str(element).lower()+"_top_"+str(int(top_percentage*100))+"_overlap.json")
            self.write_json(results_path, json_overlap)
                
    def hypothesis_3(self, ranked_quads, embeddings, normalization_scores = None):        
        entity_measures = {}
        print("Testing hypothesis 3.")

        for quad in ranked_quads:
            if quad["HEAD"] is "0" or quad["TAIL"] is "0":
                continue

            entity_n = quad["HEAD"]
            entity_m = quad["TAIL"]
            key = entity_n+";"+entity_m

            if key not in entity_measures.keys():
                entity_measures[key] = {"ENTITY_N": entity_n, "ENTITY_M": entity_m, "FACTS": 0, "RANK": Measure()}
            
            ranks = {}
            for embedding in embeddings:
                if embedding == "TFLEX":
                    if not (quad["TAIL"] == "0" or quad["TIME"] == "0"):
                        continue
                
                ranks[embedding] = int(float(quad["RANK"][embedding]))
            entity_measures[key]["RANK"].update(ranks)
            entity_measures[key]["FACTS"] += 1
        
        for key in entity_measures.keys():
            entity_measures[key]["RANK"].normalize()
        
        for key in entity_measures.keys():
            entity_n = entity_measures[key]["ENTITY_N"]
            entity_m = entity_measures[key]["ENTITY_M"]
            other_key = entity_m+";"+entity_n
            if other_key in entity_measures.keys():
                if entity_measures[key]["FACTS"] >= 5 and entity_measures[other_key]["FACTS"] >= 5:
                    entity_measures[key]["DIFFERENCE"] = {}
                    for embedding in embeddings:
                        entity_measures[key]["DIFFERENCE"][embedding] = entity_measures[other_key]["RANK"].mrr[embedding] - entity_measures[key]["RANK"].mrr[embedding]
                
        json_output = []
        for i, key in enumerate(entity_measures.keys()):
            json_output.append(deepcopy(entity_measures[key]))
            json_output[i]["RANK"] = json_output[i]["RANK"].as_dict()

        json_output_normalized = []
        if normalization_scores is not None:
            for i, key in enumerate(entity_measures.keys()):
                entity_measures[key]["RANK"].normalize_to(normalization_scores)
            
            for i, key in enumerate(entity_measures.keys()):
                entity_n = entity_measures[key]["ENTITY_N"]
                entity_m = entity_measures[key]["ENTITY_M"]
                other_key = entity_m+";"+entity_n
                if other_key in entity_measures.keys():
                    if entity_measures[key]["FACTS"] >= 5 and entity_measures[other_key]["FACTS"] >= 5:
                        entity_measures[key]["DIFFERENCE"] = {}
                        for embedding in embeddings:
                            entity_measures[key]["DIFFERENCE"][embedding] = entity_measures[other_key]["RANK"].mrr[embedding] - entity_measures[key]["RANK"].mrr[embedding]                  

                json_output_normalized.append(deepcopy(entity_measures[key]))
                json_output_normalized[i]["RANK"] = json_output_normalized[i]["RANK"].as_dict()

        json_output.sort(key=lambda val: val["FACTS"], reverse=True)
        results_path = os.path.join(self.params.base_directory, "result", self.params.dataset, "hypothesis_3", "hypothesis_3.json")
        self.write_json(results_path, json_output)

        if normalization_scores is not None:
            json_output_normalized.sort(key=lambda val: val["FACTS"], reverse=True)
            results_path = os.path.join(self.params.base_directory, "result", self.params.dataset, "hypothesis_3", "hypothesis_3_normalized.json")
            self.write_json(results_path, json_output_normalized)
    
    def entity_MRR_Sort(self, entity_scores, method_name):
        
        #print(entity_scores[1]['MEASURE']['DE_TransE'])
        sortedlist = sorted(entity_scores, key=lambda d: d['MEASURE']['DE_TransE']['MRR'], reverse=True)
        #for i in range(0,30):
        #    print(sortedlist[i]['MEASURE'][method_name])
        return sortedlist

    def get_Top_5_Elements(self):

        
        return
    def no_of_elements(self, dataset):
        entities = {}
        relations = {}
        timestamps = {}

        for line in dataset.values:
            if line[0] not in entities.keys():
                entities[line[0]] = 0
            if line[1] not in relations.keys():
                relations[line[1]] = 0
            if line[2] not in entities.keys():
                entities[line[2]] = 0
            if line[3] not in timestamps.keys():
                timestamps[line[3]] = 0

            entities[line[0]] += 1
            relations[line[1]] += 1
            entities[line[2]] += 1
            timestamps[line[3]] += 1
        
        entities_json = []
        relations_json = []
        timestamps_json = []

        for key in entities.keys():
            entities_json.append({"ENTITY": key, "COUNT": entities[key]})
        for key in relations.keys():
            relations_json.append({"RELATION": key, "COUNT": relations[key]})
        for key in timestamps.keys():
            timestamps_json.append({"TIMESTAMP": key, "COUNT": timestamps[key]})
        
        entities_json.sort(key=lambda val: val["COUNT"], reverse=True)
        results_path = os.path.join(self.params.base_directory, "result", self.params.dataset, "no_of_elements", "train_entities.json")
        self.write_json(results_path, entities_json)

        relations_json.sort(key=lambda val: val["COUNT"], reverse=True)
        results_path = os.path.join(self.params.base_directory, "result", self.params.dataset, "no_of_elements", "train_relations.json")
        self.write_json(results_path, relations_json)

        timestamps_json.sort(key=lambda val: val["COUNT"], reverse=True)
        results_path = os.path.join(self.params.base_directory, "result", self.params.dataset, "no_of_elements", "train_timestamps.json")
        self.write_json(results_path, timestamps_json)
                

    def run(self):
        embeddings = ["DE_TransE", "DE_SimplE", "DE_DistMult", "TERO", "ATISE", "TFLEX"]

        ranks_path = os.path.join(self.params.base_directory, "result", self.params.dataset, "ranked_quads.json")
        ranked_quads = self.read_json(ranks_path)
        
        learn_path = os.path.join(self.params.base_directory, "dataprepare", "corruptedquadruple", self.params.dataset, "train.txt")
        dataset = self.read_csv(learn_path)

        #self.calculate_overall_scores(ranked_quads, embeddings)

        overall_scores_path = os.path.join(self.params.base_directory, "result", self.params.dataset, "overall_scores.json")        
        overall_scores = self.read_json(overall_scores_path)

        entities_path = os.path.join(self.params.base_directory, "result", self.params.dataset, "hypothesis_2", "entity.json")        
        entity_scores = self.read_json(entities_path)

        #self.no_of_elements(dataset)
        #self.hypothesis_1(ranked_quads, embeddings, overall_scores)
        #self.hypothesis_2(ranked_quads, embeddings, overall_scores)
        self.hypothesis_2_top_x(embeddings)
        #self.hypothesis_3(ranked_quads, embeddings, overall_scores)
