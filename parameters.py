import os


class Parameters:
    def __init__(self, args):
        self.learning_rate = 0.001
        self.se_prop = 0.9
        self.emb_dim = 100
        self.dropout_probability = 0.2
        self.neg_ratio = 2
        self.batch_size = 512

        self.dataset = args.dataset
        self.task = args.task
        self.embedding = args.embedding
        self.base_directory = os.getcwd()
        self.dataloader = None
        self.device = None
        self.model_path = None
