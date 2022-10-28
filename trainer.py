import os

import torch
import torch.nn as nn

class Trainer:
    def __init__(self, params):
        self.params = params

    def train(self):
        self.params.model.train()

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.params.model.parameters(), self.params.model.learning_rate)

        for epoch in range(1, 3):
            print(" --------- Epoch: " + str(epoch) + " --------- ")

            total_loss = 0.0

            # TODO: Implement batch loading from dataset, see https://github.com/BorealisAI/de-simple/blob/master/trainer.py

            heads, rels, tails, years, months, days = self.params.dataset.get_all_data()

            scores = self.params.model(heads, rels, tails, years, months, days)
            loss = loss_fn(scores)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            print("Loss in epoch " + str(epoch) + ": " + str(total_loss))

            self.save_model(epoch)

    def save_model(self, checkpoint):
        print("Saving the model")

        directory = os.path.join(self.params.base_directory, "models", self.params.model_name, self.params.dataset)
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory, self.params.model_name + "_" + str(checkpoint) + ".checkpoint")

        torch.save(self.params.model, file_path)


