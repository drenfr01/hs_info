import torch.nn as nn


class SkipGram(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(SkipGram, self).__init__()
        # YOUR CODE HERE
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.linear = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        probs = None
        x = x.squeeze()
        embeddings = self.embedding(x)
        embeddings = nn.Sigmoid()(embeddings)
        compressions = self.linear(embeddings)
        probs = nn.LogSoftmax()(compressions)

        return probs
