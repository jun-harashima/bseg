import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random


torch.manual_seed(1)


class Model(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size,
                 batch_size=1):
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self._init_hidden()

    def _init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, self.batch_size, self.hidden_dim),
                torch.zeros(1, self.batch_size, self.hidden_dim))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

    def train(self, dataset):
        loss_function = nn.NLLLoss()
        optimizer = optim.SGD(self.parameters(), lr=0.1)
        for epoch in range(10):
            print(epoch)
            for X, Y in random.shuffle(self._split(dataset)):
                # Step 1. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                self.zero_grad()

                # Also, we need to clear out the hidden state of the LSTM,
                # detaching it from its history on the last instance.
                self.hidden = self._init_hidden()

                X, Y = self._sort(X, Y)
                X = self._pad(X, dataset.word_to_index["PAD"])
                Y = self._pad(Y, dataset.tag_to_index["PAD"])

                # Step 2. Run our forward pass.
                X = self(X)

                # Step 3. Compute the loss, gradients, and update the
                # parameters by calling optimizer.step()
                loss = loss_function(X, Y)
                loss.backward()
                optimizer.step()

    def _split(self, dataset):
        return zip(zip(*[iter(dataset.X)]*self.batch_size),
                   zip(*[iter(dataset.Y)]*self.batch_size))

    def _sort(self, X, Y):
        X = sorted(X, key=lambda x: -len(x))
        Y = sorted(Y, key=lambda y: -len(y))
        return X, Y

    def _pad(self, Z, pad_index):
        lengths = [len(z) for z in Z]
        for i, z in enumerate(Z):
            Z[i] = z + [pad_index] * (max(lengths) - len(Z[i]))
        return Z
