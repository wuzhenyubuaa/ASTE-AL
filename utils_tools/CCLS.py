import math
import torch
from torch.optim.lr_scheduler import _LRScheduler

class CyclicCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super(CyclicCosineAnnealingLR, self).__init__(optimizer, last_epoch)
    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2 for base_lr in self.base_lrs]

# Example usage:
if __name__ == "__main__":
    model = torch.nn.Linear(10, 1)  # example model
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    # Total number of epochs
    total_epochs = 100
    # Create the scheduler
    scheduler = CyclicCosineAnnealingLR(optimizer, T_max=total_epochs)

    for epoch in range(total_epochs):
        # Training code here
        optimizer.step()
        # Update the learning rate
        scheduler.step()
        # Print the current learning rate
        print(f"Epoch {epoch + 1}: learning rate = {scheduler.get_lr()}")
