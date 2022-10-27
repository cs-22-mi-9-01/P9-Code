import torch
import torch.nn as nn

class Trainer:
    def __init__(self, params):
        self.params = params

    def train(self):
        size = self.params.dataloader.dataset.__len__()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.params.model.parameters(), self.params.model.learning_rate)

        self.params.model.train()
        for batch, element in enumerate(self.params.dataloader):
            element = element.to(self.params.device)

            # Compute prediction error
            pred = self.params.model(element)
            loss = loss_fn(pred, element)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(element)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


