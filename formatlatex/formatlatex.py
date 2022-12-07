
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
                max_val = 0.0

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

    def format_hypothesis_3(self):
        for normalized in ["", "_normalized"]:
            input_path = os.path.join(self.params.base_directory, "result", self.params.dataset, "hypothesis_3", "hypothesis_3" + normalized + ".json")
            output_path = os.path.join(self.params.base_directory, "formatlatex", "result", "hypothesis_3", "hypothesis_3" + normalized + ".json")
            
            input = self.read_json(input_path)

            min_val = 100.0
            max_val = 0.0

            result = \
            "\n" + \
            r"\begin{tabular}{r|l|RRRRRR}" + "\n" +\
            r"entity 1 &" + "\n" +\
            r"entity 2 &" + "\n" +\
            r"\multicolumn{1}{c} {DE-T} &" + "\n" +\
            r"\multicolumn{1}{c} {DE-D} &" + "\n" +\
            r"\multicolumn{1}{c} {DE-S} &" + "\n" +\
            r"\multicolumn{1}{c} {ATiSE} &" + "\n" +\
            r"\multicolumn{1}{c} {TeRo} &" + "\n" +\
            r"\multicolumn{1}{c} {TFLEX}\\ \hline" + "\n"
            for i in range(0, 5):
                result += input[i]["ENTITY_N"] + r" & " + input[i]["ENTITY_M"]
                for embedding in input[i]["DIFFERENCE"].keys():
                    val = round(input[i]["DIFFERENCE"][embedding], 2)

                    if val < min_val:
                        min_val = val
                    if val > max_val:
                        max_val = val

                    result += r" & " + str(val)
                result += r"\\" + "\n"
            result += \
            r"\end{tabular}" + "\n"
            result = "\n" + r"\renewcommand{\MinNumber}{" + str(min_val) + r"}%" + "\n" +\
            r"\renewcommand{\MaxNumber}{" + str(max_val) + r"}%" + "\n" + result

            self.write(output_path, result)

    def format(self):
        self.format_hypothesis_2()
