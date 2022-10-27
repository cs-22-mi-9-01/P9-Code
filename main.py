
import argparse
import torch

from torch.utils.data import DataLoader

from trainer import Trainer

def learn(args):
    trainer = Trainer(args)
    trainer.train()

def main():
    desc = 'Temporal KG Completion methods'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-dataset', help='Dataset', type=str, default='icews14', choices=['icews14', 'icews05-15', 'gdelt'])
    parser.add_argument('-model', help='Model', type=str, default='DE_DistMult', choices=['DE_DistMult', 'DE_TransE', 'DE_SimplE'])
    parser.add_argument('-bsize', help='Batch size', type=int, default=64, choices = [64])
    parser.add_argument('-task', help='Task', type=str, default='learn', choices=['learn', 'answer'])
    parser.add_argument('-embedding', help='Embedding', type=str, default='DE-TransE', choices=['DE-TransE', 'DE-SimplE'])

    args = parser.parse_args()

    # Create data loaders.
    dataloader = DataLoader(training_data, batch_size=batch_size)

    match args.task:
        case 'learn':
            learn(args)
        case 'answer':
            print("test 2")

if __name__ == '__main__':
    main()


