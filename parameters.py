import os


class Parameters:
    def __init__(self, args):
        self.dataset = args.dataset
        self.batch_size = None
        self.task = args.task
        self.embedding = args.embedding
        self.base_directory = os.getcwd()
        self.dataloader = None
        self.device = None
        self.model_path = None
