
import argparse
from parameters import Parameters


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset', help='Dataset', type=str, default='icews14', choices=['icews14', 'icews05-15', 'gdelt'])
    parser.add_argument('-task', help='Task', type=str, default='learn', choices=['learn', 'test', 'answer'])
    parser.add_argument('-embedding', help='Embedding', type=str, default='DE-TransE', choices=['DE-TransE', 'DE-SimplE'])
    parser.add_argument('-model_path', help='Path to model', type=str, default='path')

    args = parser.parse_args()
    params = Parameters(args)


if __name__ == '__main__':
    main()


