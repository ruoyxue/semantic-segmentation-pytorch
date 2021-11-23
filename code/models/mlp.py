import torch.nn as nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

        def init_weights(m: nn.Module):
            if type(m) == nn.Linear:
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        self.sequence.apply(init_weights)

    def forward(self, x):
        pred = self.sequence(x)
        return pred

