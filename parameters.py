import os


class Parameters:
    def __init__(self, args):
        self.dataset = args.dataset
        self.embedding = args.embedding
        self.base_directory = os.getcwd()
