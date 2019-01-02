import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as U
import torch.optim as optim
import random


torch.manual_seed(1)


class WordBasedTagger(nn.Module):

    EPOCH_NUM = 100

    # For simplicity, use the same pad_index (usually 0) for words and tags
    def __init__(self, embedding_dim, hidden_dim, word_to_index, tag_to_index,
                 pad_index=0, batch_size=16):
        super(WordBasedTagger, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = len(word_to_index)
        self.tagset_size = len(tag_to_index)
        self.batch_size = batch_size
        self.pad_index = pad_index
        self.use_cuda = self._init_use_cuda()
        self.device = self._init_device()
        self.embeddings = self._init_embeddings()
        self.lstm = self._init_lstm()
        self.hidden2tag = self._init_hidden2tag()

    def _init_use_cuda(self):
        return torch.cuda.is_available()

    def _init_device(self):
        return torch.device('cuda' if self.use_cuda else 'cpu')

    def _init_embeddings(self):
        embeddings = nn.Embedding(self.vocab_size, self.embedding_dim,
                                  self.pad_index)
        return embeddings.cuda() if self.use_cuda else embeddings

    def _init_lstm(self):
        lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, bidirectional=True)
        return lstm.cuda() if self.use_cuda else lstm

    def _init_hidden2tag(self):
        hidden2tag = nn.Linear(self.hidden_dim * 2, self.tagset_size)
        return hidden2tag.cuda() if self.use_cuda else hidden2tag

    def _init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        zeros = torch.zeros(2, self.batch_size, self.hidden_dim,
                            device=self.device)
        return (zeros, zeros)

    def forward(self, X, lengths):
        X = self._embed(X)
        X = self._forward(X, lengths)
        return X

    def _forward(self, X, lengths):
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

    def train(self, train_set, dev_set=None):
        optimizer = optim.SGD(self.parameters(), lr=0.1)
        for epoch in range(1, self.EPOCH_NUM + 1):
            batches = self._split(train_set)
            random.shuffle(batches)
            self._train(optimizer, batches, epoch, dev_set)

    def _train(self, optimizer, batches, epoch, dev_set):
        loss_sum = 0
        for X, Y in batches:
            self.zero_grad()
            self.hidden = self._init_hidden()
            X, lengths, _ = self._tensorize(X)
            Y, lengths, _ = self._tensorize(Y)
            Y_hat = self(X, lengths)
            loss = self._calc_cross_entropy(Y_hat, Y)
            loss.backward()
            optimizer.step()
            loss_sum += loss
        print('epoch {:>3}\tloss {:6.2f}'.format(epoch, loss_sum))
        if dev_set is not None and epoch % 10 == 0:
            self.eval(dev_set)

    def test(self, test_set):
        results = []
        batches = self._split(test_set)
        for X, _ in batches:
            self.hidden = self._init_hidden()
            X, lengths, indices = self._tensorize(X)
            mask = (X > 0).long()
            Y_hat = self(X, lengths)
            self._extend(results, Y_hat, mask, indices)
        return results

    def eval(self, test_set):
        ok = 0
        ng = 0
        results = self.test(test_set)
        for y, y_hat in zip(test_set.Y, results):
            for _y, _y_hat in zip(y, y_hat):
                if _y == _y_hat:
                    ok += 1
                else:
                    ng += 1
        print(ok / (ok + ng))

    def _split(self, dataset):
        if len(dataset.X) < self.batch_size:
            self.batch_size = len(dataset.X)
        return list(zip(zip(*[iter(dataset.X)]*self.batch_size),
                        zip(*[iter(dataset.Y)]*self.batch_size)))

    def _tensorize(self, Z):
        Z, indices_before_sort = self._sort(Z)
        lengths_after_sort = [len(z) for z in Z]
        Z = self._pad(Z, lengths_after_sort, self.pad_index)
        Z = torch.tensor(Z, device=self.device)
        return Z, lengths_after_sort, indices_before_sort

    def _sort(self, Z):
        indices, Z = zip(*sorted(enumerate(Z), key=lambda z: -len(z[1])))
        return list(Z), list(indices)

    def _pad(self, Z, lengths):
        for i, z in enumerate(Z):
            Z[i] = z + [self.pad_index] * (max(lengths) - len(Z[i]))
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

    def _extend(self, results, Y_hat, mask, indices):
        _, __results = Y_hat.max(-1)
        __results = __results * mask
        _results = [None] * len(Y_hat)
        for i, index in enumerate(indices):
            _results[index] = [y for y in __results[i].tolist() if y != 0]
        results.extend(_results)
