
import argparse
import torch

from dataset import KnowledgeGraphDataset
from parameters import Parameters
from tester import Tester
from trainer import Trainer


def main():
    desc = 'Temporal KG Completion methods'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-dataset', help='Dataset', type=str, default='icews14', choices=['icews14', 'icews05-15', 'gdelt'])
    parser.add_argument('-task', help='Task', type=str, default='learn', choices=['learn', 'test', 'answer'])

    # Learn-specific arguments
    parser.add_argument('-embedding', help='Embedding', type=str, default='DE-TransE', choices=['DE-TransE', 'DE-SimplE'])

    # Test-specific arguments
    parser.add_argument('-model_path', help='Path to model', type=str, default='path')

    args = parser.parse_args()
    params = Parameters(args)

    # TEST BLOCK
    params.task = "test"
    params.model_path = r"C://Users//Jeppe//Documents//Unistuff//Master//P9-Code//models//DE_TransE//icews14//DE_TransE_100.model"
    # TEST BLOCK END

    # Create dataset
    dataset = KnowledgeGraphDataset(params)
    params.dataset_name = params.dataset
    params.dataset = dataset

    # Assign device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    params.device = device

    #print(f"Model structure: {model}\n\n")
    #for name, param in model.named_parameters():
    #    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

    match params.task:
        case 'learn':
            # Learn model based on embedding
            trainer = Trainer(params)
            trainer.train()
        case 'test':
            # Test the model with a test dataset
            tester = Tester(params)
            tester.test()
        case 'answer':
            print("test 2")

if __name__ == '__main__':
    main()


