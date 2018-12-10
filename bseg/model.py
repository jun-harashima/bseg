import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as U
import torch.optim as optim
import random


torch.manual_seed(1)


class Model(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size,
                 batch_size=1, padding_idx=0):
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.tagset_size = tagset_size
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self._init_hidden()

    def _init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, self.batch_size, self.hidden_dim),
                torch.zeros(1, self.batch_size, self.hidden_dim))

    def forward(self, X, lengths, pad_index):
        X = self._pad(X, lengths, pad_index)
        X = self._embed(torch.tensor(X))
        X = self._pack(X, lengths)
        X, self.hidden = self._lstm(X)
        X, _ = self._unpack(X)
        X = X.contiguous().view(-1, X.shape[2])
        # Note that hidden2tag returns values also for padded elements. We
        # ignore them when computing our loss.
        X = self.hidden2tag(X)
        X = F.log_softmax(X, dim=1)
        X = X.view(self.batch_size, lengths[0], self.tagset_size)
        return X

    def train(self, dataset):
        optimizer = optim.SGD(self.parameters(), lr=0.1)
        for epoch in range(10):
            batches = self._split(dataset)
            random.shuffle(batches)
            accumulated_loss = 0
            for X, Y in batches:
                self.zero_grad()
                self.hidden = self._init_hidden()

                X = self._sort(X)
                lengths = [len(x) for x in X]
                X = self(X, lengths, dataset.word_to_index["<PAD>"])

                Y = self._sort(Y)
                Y = self._pad(Y, lengths, dataset.tag_to_index["<PAD>"])
                Y = torch.tensor(Y)

                loss = self._calc_cross_entropy(X, Y)
                loss.backward()
                optimizer.step()
                accumulated_loss += loss
            print("epoch: {} loss: {}".format(epoch, accumulated_loss))

    def _split(self, dataset):
        return list(zip(zip(*[iter(dataset.X)]*self.batch_size),
                        zip(*[iter(dataset.Y)]*self.batch_size)))

    def _sort(self, Z):
        return sorted(Z, key=lambda z: -len(z))

    def _pad(self, Z, lengths, pad_index):
        for i, z in enumerate(Z):
            Z[i] = z + [pad_index] * (max(lengths) - len(Z[i]))
        return Z

    def _embed(self, X):
        return self.embeddings(X)

    def _pack(self, X, lengths):
        return U.rnn.pack_padded_sequence(X, lengths, batch_first=True)

    def _lstm(self, X):
        return self.lstm(X, self.hidden)

    def _unpack(self, X):
        return U.rnn.pad_packed_sequence(X, batch_first=True)

    def _calc_cross_entropy(self, X, Y):
        # X = [[[-2.02, -1.97, -1.66, ...], -> [[-2.02, -1.97, -1.66, ...],
        #       [-2.06, -1.85, -1.70, ...]]     [-2.06, -1.85, -1.70, ...],
        #      [[-2.12, -1.91, -1.65, ...],     [-2.12, -1.91, -1.65, ...],
        #       [-2.16, -1.85, -1.66, ...]]])   [-2.16, -1.85, -1.66, ...]])
        X = X.view(-1, self.tagset_size)
        # Y = [[1, 2], [1, 0]] -> [1, 2, 1, 0]
        Y = Y.view(-1)
        # X = [-1.97, -1.70, -1.91, -2.16]
        X = X[range(X.shape[0]), Y]
        # mask = [1., 1., 1., 0.] (for <PAD>)
        mask = (Y > 0).float()
        # X = [-1.97, -1.70, -1.91, -0.00]
        X = X * mask
        # token_num = 3
        token_num = int(torch.sum(mask))
        # -1 * (-1.97 + -1.70 + -1.91 + -0.00) / 3
        return -torch.sum(X) / token_num
