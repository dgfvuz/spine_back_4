import torch
import torch.nn as nn

class StackHourglassLoss(nn.Module):
    def __init__(self):
        super(StackHourglassLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, outputs, target):
        loss = 0
        for output in outputs:
            loss += self.mse_loss(output, target)
        return loss