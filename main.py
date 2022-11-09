
import argparse
import os
import json
import sys

from loader import Loader
from parameters import Parameters
from ranker import Ranker
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset', type=str, default='icews14', choices=['icews14', 'icews05-15', 'gdelt'])
    parser.add_argument('-embedding', type=str, default='all', choices=['DE-TransE', 'DE-SimplE'])

    args = parser.parse_args()
    params = Parameters(args)

    quads_path = os.path.join(params.base_directory, "data", params.dataset, "corrupted_quads.json")

    in_file = open(quads_path, "r")
    ranked_quads = json.load(in_file)
    in_file.close()

    if params.embedding == "all":
        embeddings = ["DE_TransE"]
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
    out_file = open(results_path, "w")
    json.dump(ranked_quads, out_file, indent=4)
    out_file.close()


if __name__ == '__main__':
    main()


