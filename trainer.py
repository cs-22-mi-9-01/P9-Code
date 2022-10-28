import os
import torch

class Trainer:
    def __init__(self, params):
        self.params = params

        self.epochs = 500
        self.save = False
        self.save_every = 500

    def train(self):
        self.params.model.train()

        loss_fn = self.params.model.loss_function
        optimizer = torch.optim.Adam(self.params.model.parameters(), self.params.model.learning_rate)

        for epoch in range(1, self.epochs + 1):
            print("--------- Epoch: " + str(epoch) + " ---------")

            total_loss = 0.0

            # TODO: Implement batch loading from dataset, see https://github.com/BorealisAI/de-simple/blob/master/trainer.py

            heads, rels, tails, years, months, days = self.params.dataset.get_all_data()
            scores = self.params.model(heads, rels, tails, years, months, days)

            target = torch.zeros(heads.shape[0]).to(self.params.device)
            loss = loss_fn(scores, target)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()

            print("Loss in epoch " + str(epoch) + ": " + str(total_loss))

            if self.save and epoch % self.save_every == 0:
                self.save_model(epoch)

    def save_model(self, checkpoint):
        directory = os.path.join(self.params.base_directory, "models", self.params.model_name, self.params.dataset_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory, self.params.model_name + "_" + str(checkpoint) + ".model")

        print("Saving the model (" + file_path + ")")
        torch.save(self.params.model, file_path)


