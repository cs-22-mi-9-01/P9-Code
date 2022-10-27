import torch
import torch.nn as nn

class Trainer:
    def __init__(self, params):
        self.params = params

    def train(self):
        size = len(self.params.dataloader.dataset)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.params.model.parameters(), self.params.model.learning_rate)

        self.params.model.train()
        for batch, (h, r, t, time) in enumerate(self.params.dataloader):
            (h, r, t, time) = h.to(self.params.device), r.to(self.params.device), t.to(self.params.device), time.to(self.params.device)

            # Compute prediction error
            pred = self.params.model(heads, rels, tails, times)
            loss = loss_fn(pred, )

            # Backpropagation
            optimizer.zero_grad()
            #loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * size
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


