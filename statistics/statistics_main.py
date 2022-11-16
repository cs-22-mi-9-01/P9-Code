
import os
import json

from statistics.measure import Measure


def hypothesis_1(ranked_quads):
    for element_type in ["HEAD", "RELATION", "TAIL", "TIME"]:
        print("Rank of question tuples when " + str(element_type) + " is the answer element:")
        
        measure = Measure()
        num_facts = 0

        for quad in ranked_quads:
            if quad[element_type] is not "0":
                continue

            num_facts += 1
            for embedding in quad["RANK"].keys():
                measure.update(embedding, int(quad["RANK"][embedding]))
        
        measure.normalize()
        measure.print()

def hypothesis_2(ranked_quads, embeddings):
    for element_type in ["HEAD", "TAIL"]:
        print("Testing hypothesis 2 on " + str(element_type) + " relations:")

        for embedding in embeddings:
            entity_measures = {}
            for quad in ranked_quads:
                if quad[element_type] is "0":
                    continue

                if quad[element_type] not in entity_measures.keys():
                    entity_measures[quad[element_type]] = Measure()
                
                entity_measures[quad[element_type]].update(embedding, int(quad["RANK"][embedding]))
            
            measure_pairs = []
            for entity in entity_measures.keys():
                entity_measures[entity].normalize()
                measure_pairs.append((entity, entity_measures[entity]))
            measure_pairs.sort(key=lambda pair: pair[1].mr[embedding])

            max_num_of_ent = 10
            for (entity, measure) in measure_pairs:
                if measure.num_facts < 10:
                    continue

                max_num_of_ent -= 1
                if max_num_of_ent == 0:
                    break

                print("Entity: "+str(entity)+":")
                measure.print()

def statistics_main(params):
    ranks_path = os.path.join(params.base_directory, "result", params.dataset, "ranked_quads.json")
    
    in_file = open(ranks_path, "r", encoding="utf8")
    ranked_quads = json.load(in_file)
    in_file.close()

    embeddings = ranked_quads[0]["RANK"].keys()

    #hypothesis_1(ranked_quads)
    hypothesis_2(ranked_quads, embeddings)
    
        


        
