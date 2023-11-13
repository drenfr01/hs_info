import torch.nn as nn


class CBOW(nn.Module):
    """Pytorch definition of CBOW model"""

    def __init__(self, vocab_size, embed_size):
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.linear = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.sum(x, dim=1)
        x = nn.Sigmoid()(x)
        x = self.linear(x)
        probs = nn.LogSoftmax()(x)
        return probs
