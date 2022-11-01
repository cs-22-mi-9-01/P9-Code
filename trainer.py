import os
import torch

from embeddings.de_transe import DETransE
from scripts import split_facts


class Trainer:
    def __init__(self, params):
        self.params = params
        self.model = DETransE(params).to(self.params.device)
        self.params.model = self.model

        self.epochs = 500
        self.save = True
        self.save_every = 50

    def train(self):
        self.params.model.train()

        loss_fn = self.params.model.loss_function
        optimizer = torch.optim.Adam(self.params.model.parameters(), self.params.model.learning_rate)

        for epoch in range(1, self.epochs + 1):
            print("--------- Epoch: " + str(epoch) + " ---------")
            last_batch = False
            total_loss = 0.0

            self.params.dataset.reset_batches()
            while not last_batch:
                facts, neg_samples = self.params.dataset.get_next_batch()
                batch_no, last_batch_no = self.params.dataset.get_batch_no()
                print("Batch number " + str(batch_no) + "/" + str(last_batch_no))
                if batch_no == last_batch_no:
                    last_batch = True

                heads, rels, tails, years, months, days = split_facts(facts)
                scores = self.params.model(heads, rels, tails, years, months, days)
                heads, rels, tails, years, months, days = split_facts(neg_samples)
                scores_neg = self.params.model(heads, rels, tails, years, months, days)

                target = torch.zeros(heads.shape[0]).to(self.params.device)
                loss = loss_fn(scores, scores_neg, target)
                loss.backward()
                total_loss += loss.item()
                optimizer.step()

            print("Loss in epoch " + str(epoch) + ": " + str(total_loss))

            if self.save and epoch % self.save_every == 0:
                self.save_model(epoch)

    def save_model(self, checkpoint):
        directory = os.path.join(self.params.base_directory, "models", self.params.model.name, self.params.dataset_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory, self.params.model.name + "_" + str(checkpoint) + ".model")

        print("Saving the model (" + file_path + ")")
        torch.save(self.params.model, file_path)


