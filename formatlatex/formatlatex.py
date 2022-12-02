
import json
from pathlib import Path

class FormatLatex():
    def __init__(self, input_path, output_path) -> None:
        self.input = self.read_json(input_path)
        self.output_path = output_path

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
        min_val = 100.0
        max_val = 0.0

        result = \
        "\n" + \
        r"\begin{tabular}{ r R R R R R R}" + "\n" +\
        r"\multicolumn{1}{c} {} &" + "\n" +\
        r"\multicolumn{1}{c} {DE-T} &" + "\n" +\
        r"\multicolumn{1}{c} {DE-D} &" + "\n" +\
        r"\multicolumn{1}{c} {DE-S} &" + "\n" +\
        r"\multicolumn{1}{c} {ATiSE} &" + "\n" +\
        r"\multicolumn{1}{c} {TeRo} &" + "\n" +\
        r"\multicolumn{1}{c} {TFLEX}\\" + "\n"
        for i in range(0, 5):
            result += self.get_entity(self.input[i])
            for embedding in self.input[i]["MEASURE"].values():
                val = round(embedding["MRR"], 2)

                if val < min_val:
                    min_val = val
                if val > max_val:
                    max_val = val

                result += r" & " + str(val)
            result += r"\\" + "\n"
        result += \
        r"\end{tabular}" + "\n" +\
        "\n"
        result = r"\renewcommand{\MinNumber}{" + str(0) + r"{%" + "\n" +\
        r"\renewcommand{\MaxNumber}{" + str(max_val) + r"{%" + "\n" + result

        return result

    def format(self):
        text = self.format_hypothesis_2()
        self.write(self.output_path, text)
