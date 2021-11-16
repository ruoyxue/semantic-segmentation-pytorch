import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.sequence = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

        def init_weights(m : nn.Module):
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, std=0.01)
        self.sequence.apply(init_weights)

    def forward(self, X):
        pred = self.sequence(X)
        return pred

def get_mlp():
    return MLP()
