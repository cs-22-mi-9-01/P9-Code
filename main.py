
import argparse
import torch

from dataset import KnowledgeGraphDataset
from torch.utils.data import DataLoader
from parameters import Parameters
from embeddings.de_transe import DETransE
from trainer import Trainer


def main():
    desc = 'Temporal KG Completion methods'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-dataset', help='Dataset', type=str, default='icews14', choices=['icews14', 'icews05-15', 'gdelt'])
    parser.add_argument('-model', help='Model', type=str, default='DE_DistMult', choices=['DE_DistMult', 'DE_TransE', 'DE_SimplE'])
    parser.add_argument('-bsize', help='Batch size', type=int, default=64, choices = [64])
    parser.add_argument('-task', help='Task', type=str, default='learn', choices=['learn', 'answer'])
    parser.add_argument('-embedding', help='Embedding', type=str, default='DE-TransE', choices=['DE-TransE', 'DE-SimplE'])

    args = parser.parse_args()
    params = Parameters(args)

    # Create data loaders
    dataset = KnowledgeGraphDataset(params)
    dataloader = DataLoader(dataset, batch_size=params.batch_size)
    params.dataloader = dataloader

    # Assign device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    params.device = device

    # Use model
    model = DETransE(params).to(device)
    params.model = model

    match args.task:
        case 'learn':
            trainer = Trainer(params)
            trainer.train()
        case 'answer':
            print("test 2")

if __name__ == '__main__':
    main()


