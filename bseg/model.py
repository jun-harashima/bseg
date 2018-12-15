import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as U
import torch.optim as optim
import random


torch.manual_seed(1)


class Model(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, word_to_index, tag_to_index,
                 word_pad_index=0, tag_pad_index=0, batch_size=32):
        super(Model, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = len(word_to_index)
        self.tagset_size = len(tag_to_index)
        self.word_pad_index = word_pad_index
        self.tag_pad_index = tag_pad_index
        self.batch_size = batch_size
        self.device = self._init_device()
        self.embeddings = self._init_embeddings()
        self.lstm = self._init_lstm()
        self.hidden2tag = self._init_hidden2tag()

    def _init_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _init_embeddings(self):
        embeddings = nn.Embedding(self.vocab_size, self.embedding_dim,
                                  self.word_pad_index)
        return embeddings.cuda() if torch.cuda.is_available() else embeddings

    def _init_lstm(self):
        lstm = nn.LSTM(self.embedding_dim, self.hidden_dim)
        return lstm.cuda() if torch.cuda.is_available() else lstm

    def _init_hidden2tag(self):
        hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)
        return hidden2tag.cuda() if torch.cuda.is_available() else hidden2tag

    def _init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        zeros = torch.zeros(1, self.batch_size, self.hidden_dim,
                            device=self.device)
        return (zeros, zeros)

    def forward(self, X, lengths):
        X = self._embed(X)
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
                X, lengths, _ = self._tensorize(X, self.word_pad_index)
                Y, lengths, _ = self._tensorize(Y, self.tag_pad_index)
                X = self(X, lengths)
                loss = self._calc_cross_entropy(X, Y)
                loss.backward()
                optimizer.step()
                accumulated_loss += loss
            print("epoch: {} loss: {}".format(epoch, accumulated_loss))

    def test(self, dataset):
        results = []
        batches = self._split(dataset)
        for X, _ in batches:
            self.hidden = self._init_hidden()
            X, lengths, indices = self._tensorize(X, self.word_pad_index)
            mask = (X > 0).long()
            X = self(X, lengths)
            self._append(results, X, mask, indices)
        return results

    def _split(self, dataset):
        if len(dataset.X) < self.batch_size:
            self.batch_size = len(dataset.X)
        return list(zip(zip(*[iter(dataset.X)]*self.batch_size),
                        zip(*[iter(dataset.Y)]*self.batch_size)))

    def _tensorize(self, Z, pad_index):
        Z, indices_before_sort = self._sort(Z)
        lengths_after_sort = [len(z) for z in Z]
        Z = self._pad(Z, lengths_after_sort, pad_index)
        Z = torch.tensor(Z, device=self.device)
        return Z, lengths_after_sort, indices_before_sort

    def _sort(self, Z):
        indices, Z = zip(*sorted(enumerate(Z), key=lambda z: -len(z[1])))
        return list(Z), list(indices)

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

    def _append(self, results, X, mask, indices):
        _, __results = X.max(-1)
        __results = __results * mask
        _results = [None] * len(X)
        for i, index in enumerate(indices):
            _results[index] = [y for y in __results[i].tolist() if y != 0]
        results.append(_results)
