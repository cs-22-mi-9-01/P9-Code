
import argparse
import os
import json

from loader import Loader
from parameters import Parameters
from ranker import Ranker
from pathlib import Path

from statistics.statistics import Statistics
from formatlatex.formatlatex import FormatLatex


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-task', type=str, default='statistics', choices=['statistics', 'rank', 'formatlatex'])
    parser.add_argument('-dataset', type=str, default='icews14', choices=['icews14', 'icews05-15', 'gdelt'])
    parser.add_argument('-embedding', type=str, default='all', choices=['all', 'DE_TransE', 'DE_SimplE', 'DE_DistMult', 'TERO', 'ATISE', 'TFLEX','TimePlex'])
    parser.add_argument('-add_to_result', type=bool, default=True)

    args = parser.parse_args()
    params = Parameters(args)
    
    match params.task:
        case "statistics":
            statistics = Statistics(params)
            statistics.run()
            return 0
        case "formatlatex":
            format_latex = FormatLatex(params)
            format_latex.format()
            return 0

    if not params.add_to_result:
        quads_path = os.path.join(params.base_directory, "data", params.dataset, "corrupted_quads.json")
    else:
        quads_path = os.path.join(params.base_directory, "result", params.dataset, "ranked_quads.json")

    in_file = open(quads_path, "r", encoding="utf8")
    ranked_quads = json.load(in_file)
    in_file.close()

    if params.embedding == "all":
        embeddings = ["DE_TransE", "DE_SimplE", "DE_DistMult", 'TERO', 'ATISE', 'TFLEX']
    else:
        embeddings = [params.embedding]

    for embedding in embeddings:
        model_path = os.path.join(params.base_directory, "models", embedding, params.dataset, "Model.model")
        loader = Loader(params, model_path, embedding)
        model = loader.load()

        ranker = Ranker(params, ranked_quads, model, embedding)
        ranked_quads = ranker.rank()

    results_path = os.path.join(params.base_directory, "result", params.dataset, "ranked_quads.json")

    Path(results_path).touch(exist_ok=True)
    out_file = open(results_path, "w", encoding="utf8")
    json.dump(ranked_quads, out_file, indent=4)
    out_file.close()


if __name__ == '__main__':
    main()


