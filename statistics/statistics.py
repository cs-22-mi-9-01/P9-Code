
import os
import json

from pathlib import Path

from statistics.measure import Measure

class Statistics():
    def __init__(self, params) -> None:
        self.params = params

    def hypothesis_1(self, ranked_quads, embeddings):
        for element_type in ["HEAD", "RELATION", "TAIL", "TIME"]:
            print("Rank of question tuples when " + str(element_type) + " is the answer element:")
            
            measure = Measure()

            for quad in ranked_quads:
                if quad[element_type] is not "0":
                    continue

                ranks = {}
                for embedding in embeddings:
                    ranks[embedding] = int(quad["RANK"][embedding])
                measure.update(ranks)
            
            measure.normalize()
            measure.print()

    def hypothesis_2(self, ranked_quads, embeddings):
        for element_type in ["HEAD", "TAIL"]:
            json_output = []

            print("Testing hypothesis 2 on " + str(element_type) + " relations:")

            entity_measures = {}
            for quad in ranked_quads:
                if quad[element_type] is "0":
                    continue

                if quad[element_type] not in entity_measures.keys():
                    entity_measures[quad[element_type]] = Measure()
                
                ranks = {}
                for embedding in embeddings:
                    ranks[embedding] = int(quad["RANK"][embedding])
                entity_measures[quad[element_type]].update(ranks)
            
            measure_tuples = []
            for entity in entity_measures.keys():
                entity_measures[entity].normalize()

                min_val = 1
                max_val = 0
                for embedding in embeddings:
                    if entity_measures[entity].mrr[embedding] < min_val:
                        min_val = entity_measures[entity].mrr[embedding]
                    if entity_measures[entity].mrr[embedding] > max_val:
                        max_val = entity_measures[entity].mrr[embedding]

                measure_tuples.append((entity, entity_measures[entity], max_val-min_val))
            measure_tuples.sort(key=lambda tuple: tuple[2], reverse=True)

            max_print = 2
            for (entity, measure, diff) in measure_tuples:
                json_output.append({"ENTITY": entity, "DIFFERENCE": {"MRR": diff}, "MEASURE": measure.as_dict()})

                if measure.num_facts < 10:
                    continue

                if max_print == 0:
                    break

                print("Entity: "+str(entity)+": (Difference: " + str(diff) + ")")
                measure.print()

                max_print -= 1

            results_path = os.path.join(self.params.base_directory, "result", self.params.dataset, "hypothesis_2_"+str(element_type).lower()+".json")

            Path(results_path).touch(exist_ok=True)
            out_file = open(results_path, "w", encoding="utf8")
            json.dump(json_output, out_file, indent=4)
            out_file.close()

    def run(self):
        ranks_path = os.path.join(self.params.base_directory, "result", self.params.dataset, "ranked_quads.json")
        
        in_file = open(ranks_path, "r", encoding="utf8")
        ranked_quads = json.load(in_file)
        in_file.close()

        embeddings = ranked_quads[0]["RANK"].keys()

        #self.hypothesis_1(ranked_quads, embeddings)
        self.hypothesis_2(ranked_quads, embeddings)
        
            


            
