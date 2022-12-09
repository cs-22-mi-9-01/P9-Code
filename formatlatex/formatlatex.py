
import json
from pathlib import Path
import os

class FormatLatex():
    def __init__(self, params) -> None:
        self.params = params

    def write(self, path, text):
        Path(path).touch(exist_ok=True)
        out_file = open(path, "w", encoding="utf8")
        out_file.write(text)
        out_file.close()
    
    def read_json(self, path):
        in_file = open(path, "r", encoding="utf8")
        dict = json.load(in_file)
        in_file.close()
        return dict

    def get_entity(self, measure):
        if "ENTITY" in measure.keys():
            return measure["ENTITY"]
        if "RELATION" in measure.keys():
            return measure["RELATION"]
        if "TIME" in measure.keys():
            return measure["TIME"]
    
    def round(self, val):
        return round(val, 2)

    def format_hypothesis_2(self):
        for normalized in ["", "_normalized"]:
            for element_type in ["entity", "relation", "time"]:
                input_path = os.path.join(self.params.base_directory, "result", self.params.dataset, "hypothesis_2", element_type +normalized + ".json")
                output_path = os.path.join(self.params.base_directory, "formatlatex", "result", "hypothesis_2_"+element_type+normalized+".tex")
                
                input = self.read_json(input_path)

                min_val = 100.0
                max_val = -100.0

                result = \
                "\n" + \
                r"\begin{tabular}{r|RRRRRR}" + "\n" +\
                r"\multicolumn{1}{c|} {} &" + "\n" +\
                r"\multicolumn{1}{c} {DE-T} &" + "\n" +\
                r"\multicolumn{1}{c} {DE-D} &" + "\n" +\
                r"\multicolumn{1}{c} {DE-S} &" + "\n" +\
                r"\multicolumn{1}{c} {ATiSE} &" + "\n" +\
                r"\multicolumn{1}{c} {TeRo} &" + "\n" +\
                r"\multicolumn{1}{c} {TFLEX}\\ \hline" + "\n"
                    
                for i in range(0, 5):
                    result += self.get_entity(input[i]) +\
                    r" & " + str(self.round(input[i]["MEASURE"]["DE_TransE"]["MRR"])) +\
                    r" & " + str(self.round(input[i]["MEASURE"]["DE_DistMult"]["MRR"])) +\
                    r" & " + str(self.round(input[i]["MEASURE"]["DE_SimplE"]["MRR"]) ) +\
                    r" & " + str(self.round(input[i]["MEASURE"]["ATISE"]["MRR"])) +\
                    r" & " + str(self.round(input[i]["MEASURE"]["TERO"]["MRR"])) +\
                    r" & " + str(self.round(input[i]["MEASURE"]["TFLEX"]["MRR"])) + r"\\" + "\n"

                    for embedding in input[i]["MEASURE"].values():
                        val = self.round(embedding["MRR"])

                        if val < min_val:
                            min_val = val
                        if val > max_val:
                            max_val = val

                result += \
                r"\end{tabular}" + "\n"
                result = "\n" + r"\renewcommand{\MinNumber}{" + str(min_val) + r"}%" + "\n" +\
                r"\renewcommand{\MaxNumber}{" + str(max_val) + r"}%" + "\n" + result

                self.write(output_path, result)

    def format_embedding(self, embedding):
        if embedding == 'DE_TransE':
            return 'DE-T'
        if embedding == 'DE_DistMult':
            return 'DE-D'
        if embedding == 'DE_SimplE':
            return 'DE-S'
        if embedding == 'TERO':
            return 'TeRo'
        if embedding == 'ATISE':
            return 'ATiSE'
        if embedding == 'TFLEX':
            return 'TFLEX'

    def get_overlap(self, overlaps, emb_n, emb_m):
        for o in overlaps:
            if o["EMBEDDING_N"] == emb_n and o["EMBEDDING_M"] == emb_m:
                return o["OVERLAP_TOP"]

    def to_str(self, value):
        return "{:.2f}".format(value, 2)

    def format_hypothesis_2_overlap(self):
        for element_type in ["entity", "relation", "time"]:
            input_path = os.path.join(self.params.base_directory, "result", self.params.dataset, "hypothesis_2", "top_x_overlap", element_type + "_top_50_overlap.json")
            output_path = os.path.join(self.params.base_directory, "formatlatex", "result", "hypothesis_2_" + element_type + "_top_50_overlap.tex")
            
            overlaps = self.read_json(input_path)

            min_val = 100.0
            max_val = -100.0

            result = \
            "\n" + \
            r"\begin{tabular}{r|RRRRRR}" + "\n" +\
            r"\multicolumn{1}{c|} {} &" + "\n" +\
            r"\multicolumn{1}{c} {DE-T} &" + "\n" +\
            r"\multicolumn{1}{c} {DE-D} &" + "\n" +\
            r"\multicolumn{1}{c} {DE-S} &" + "\n" +\
            r"\multicolumn{1}{c} {ATiSE} &" + "\n" +\
            r"\multicolumn{1}{c} {TeRo} &" + "\n" +\
            r"\multicolumn{1}{c} {TFLEX}\\ \hline" + "\n"
                
            for embedding_n in ['DE_TransE', 'DE_DistMult', 'DE_SimplE', 'ATISE', 'TERO', 'TFLEX']:
                result += self.format_embedding(embedding_n)
                for embedding_m in ['DE_TransE', 'DE_DistMult', 'DE_SimplE', 'ATISE', 'TERO', 'TFLEX']:
                    if embedding_n == embedding_m:
                        result += r" & \multicolumn{1}{c} {1.00}"
                    else:
                        result += r" & " + self.to_str(self.round(self.get_overlap(overlaps, embedding_n, embedding_m)))
                result += r"\\" + "\n"

            for overlap in overlaps:
                if overlap["EMBEDDING_N"] == overlap["EMBEDDING_M"]:
                    continue

                val = self.round(overlap["OVERLAP_TOP"])

                if val < min_val:
                    min_val = val
                if val > max_val:
                    max_val = val

            result += \
            r"\end{tabular}" + "\n"
            result = "\n" + r"\renewcommand{\MinNumber}{" + str(min_val) + r"}%" + "\n" +\
            r"\renewcommand{\MaxNumber}{" + str(max_val) + r"}%" + "\n" + result

            self.write(output_path, result)    

    def format_hypothesis_3(self):
        for normalized in ["", "_normalized"]:
            input_path = os.path.join(self.params.base_directory, "result", self.params.dataset, "hypothesis_3", "hypothesis_3" + normalized + ".json")
            output_path = os.path.join(self.params.base_directory, "formatlatex", "result", "hypothesis_3" + normalized + ".tex")
            
            input = self.read_json(input_path)

            min_val = 100.0
            max_val = 0.0

            result = \
            "\n" + \
            r"\begin{tabular}{r|l|SSSSSS}" + "\n" +\
            r"$e_n$ &" + "\n" +\
            r"$e_m$ &" + "\n" +\
            r"\multicolumn{1}{c} {DE-T} &" + "\n" +\
            r"\multicolumn{1}{c} {DE-D} &" + "\n" +\
            r"\multicolumn{1}{c} {DE-S} &" + "\n" +\
            r"\multicolumn{1}{c} {ATiSE} &" + "\n" +\
            r"\multicolumn{1}{c} {TeRo} &" + "\n" +\
            r"\multicolumn{1}{c} {TFLEX}\\ \hline" + "\n"
            i = 0
            num_of_rows = 0
            while num_of_rows < 5:
                if "DIFFERENCE" in input[i].keys():
                    result += input[i]["ENTITY_N"] + r" & " + input[i]["ENTITY_M"] +\
                    r" & " + str(self.round(input[i]["DIFFERENCE"]["DE_TransE"])) +\
                    r" & " + str(self.round(input[i]["DIFFERENCE"]["DE_DistMult"])) +\
                    r" & " + str(self.round(input[i]["DIFFERENCE"]["DE_SimplE"]) ) +\
                    r" & " + str(self.round(input[i]["DIFFERENCE"]["ATISE"])) +\
                    r" & " + str(self.round(input[i]["DIFFERENCE"]["TERO"])) +\
                    r" & " + str(self.round(input[i]["DIFFERENCE"]["TFLEX"])) + r"\\" + "\n"

                    for embedding in input[i]["DIFFERENCE"].keys():
                        val = self.round(input[i]["DIFFERENCE"][embedding])

                        if val < min_val:
                            min_val = val
                        if val > max_val:
                            max_val = val
                    
                    num_of_rows += 1
                i += 1
            
            result += \
            r"\end{tabular}" + "\n"
            result = "\n" + r"\renewcommand{\MinNumber}{" + str(min_val) + r"}%" + "\n" +\
            r"\renewcommand{\MaxNumber}{" + str(max_val) + r"}%" + "\n" + result

            self.write(output_path, result)

    def format_no_of_entities(self):
        for element in ["entities", "relations", "timestamps"]:
            input_path = os.path.join(self.params.base_directory, "result", self.params.dataset, "no_of_elements", "train_" + element + ".json")
            output_path = os.path.join(self.params.base_directory, "formatlatex", "result", "no_of_elements_train_" + element + ".tex")
            
            input = self.read_json(input_path)

            min_val = 999
            max_val = 0

            result = r"\addplot+ coordinates {" + "\n"
            for i, e in enumerate(input):
                val = e["COUNT"]
                result += r"   (" + str(i) + r", " + str(val) + r")" + "\n"
                
                if val < min_val:
                    min_val = val
                if val > max_val:
                    max_val = val
            
            result += r"} ;"
            result = r"% MIN VAL: " + str(min_val) + "\n" + r"% MAX VAL: " + str(max_val) + "\n\n" + result

            self.write(output_path, result)

    def format(self):
        #self.format_hypothesis_2()
        #self.format_hypothesis_3()
        #self.format_no_of_entities()
        self.format_hypothesis_2_overlap()
