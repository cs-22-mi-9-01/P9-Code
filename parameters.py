import os


class Parameters:
    def __init__(self, args):
        self.task = args.task
        self.dataset = args.dataset
        self.embedding = args.embedding

        self.base_directory = os.path.abspath(os.path.dirname(__file__))
        self.add_to_result = args.add_to_result
