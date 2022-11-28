import os


class Parameters:
    def __init__(self, args):
        self.dataset = args.dataset
        self.embedding = args.embedding
        self.base_directory = os.getcwd()
        self.add_to_result = args.add_to_result
