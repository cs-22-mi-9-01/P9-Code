import os


class Parameters:
    def __init__(self, args):
        self.dataset = args.dataset
        self.model_name = args.model
        self.batch_size = args.bsize
        self.task = args.task
        self.embedding = args.embedding
        self.base_directory = os.getcwd()
        self.dataloader = None
        self.device = None
        self.model = None
