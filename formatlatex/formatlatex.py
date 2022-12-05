
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
                    result += self.get_entity(input[i])
                    for embedding in input[i]["MEASURE"].values():
                        val = round(embedding["MRR"], 2)

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
