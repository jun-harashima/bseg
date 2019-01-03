import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as U
import torch.optim as optim
import random


torch.manual_seed(1)


class Model(nn.Module):

    EPOCH_NUM = 100

    # For simplicity, use the same pad_index for Xs[0], Xs[1], ..., and Y
    def __init__(self, embedding_dims, hidden_dims, x_set_sizes, y_set_size,
                 pad_index=0, batch_size=16):
        super(Model, self).__init__()
        self.embedding_dims = embedding_dims
        self.hidden_dims = hidden_dims
        self.x_set_sizes = x_set_sizes
        self.y_set_size = y_set_size
        self.batch_size = batch_size
        self.pad_index = pad_index
        self.use_cuda = self._init_use_cuda()
        self.device = self._init_device()
        self.embeddings = self._init_embeddings()
        self.lstm = self._init_lstm()
        self.hidden2y = self._init_hidden2y()

    def _init_use_cuda(self):
        return torch.cuda.is_available()

    def _init_device(self):
        return torch.device('cuda' if self.use_cuda else 'cpu')

    def _init_embeddings(self):
        embeddings = []
        for size, dim in zip(self.x_set_sizes, self.embedding_dims):
            embedding = nn.Embedding(size, dim, self.pad_index)
            embedding = embedding.cuda() if self.use_cuda else embedding
            embeddings.append(embedding)
        return embeddings

    def _init_lstm(self):
        lstm = nn.LSTM(sum(self.embedding_dims), sum(self.hidden_dims),
                       bidirectional=True)
        return lstm.cuda() if self.use_cuda else lstm

    def _init_hidden2y(self):
        hidden2y = nn.Linear(sum(self.hidden_dims) * 2, self.y_set_size)
        return hidden2y.cuda() if self.use_cuda else hidden2y

    def _init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        zeros = torch.zeros(2, self.batch_size, sum(self.hidden_dims),
                            device=self.device)
        return (zeros, zeros)

    def forward(self, Xs, lengths):
        Xs = self._embed(Xs)
        X = torch.cat(Xs, 2)
        X = self._pack(X, lengths)
        X, self.hidden = self._lstm(X)
        X, _ = self._unpack(X)
        X = X.contiguous().view(-1, X.shape[2])
        # Note that hidden2y returns values also for padded elements. We
        # ignore them when computing our loss.
        X = self.hidden2y(X)
        X = F.log_softmax(X, dim=1)
        X = X.view(self.batch_size, lengths[0], self.y_set_size)
        return X

    def train(self, train_set, dev_set=None):
        optimizer = optim.SGD(self.parameters(), lr=0.1)
        for epoch in range(1, self.EPOCH_NUM + 1):
            batches = self._split(train_set)
            random.shuffle(batches)
            loss_sum = 0
            for *Xs, Y in batches:
                self.zero_grad()
                self.hidden = self._init_hidden()
                for i in range(len(Xs)):
                    Xs[i], lengths, _ = self._tensorize(Xs[i])
                Y, _, _ = self._tensorize(Y)
                Y_hat = self(Xs, lengths)
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
        for *Xs, _ in batches:
            self.hidden = self._init_hidden()
            for i in range(len(Xs)):
                Xs[i], lengths, indices = self._tensorize(Xs[i])
            mask = (Xs[0] > 0).long()
            Y_hat = self(Xs, lengths)
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
        if len(dataset.Y) < self.batch_size:
            self.batch_size = len(dataset.Y)
        Zs = dataset.Xs + [dataset.Y]
        return list(zip(*[zip(*[iter(Z)]*self.batch_size) for Z in Zs]))

    def _tensorize(self, Z):
        Z, indices_before_sort = self._sort(Z)
        lengths_after_sort = [len(z) for z in Z]
        Z = self._pad(Z, lengths_after_sort)
        Z = torch.tensor(Z, device=self.device)
        return Z, lengths_after_sort, indices_before_sort

    def _sort(self, Z):
        indices, Z = zip(*sorted(enumerate(Z), key=lambda z: -len(z[1])))
        return list(Z), list(indices)

    def _pad(self, Z, lengths):
        for i, z in enumerate(Z):
            Z[i] = z + [self.pad_index] * (max(lengths) - len(Z[i]))
        return Z

    def _embed(self, Xs):
        return [self.embeddings[i](X) for i, X in enumerate(Xs)]

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
        X = X.view(-1, self.y_set_size)
        # Y = [[1, 2], [1, 0]] -> [1, 2, 1, 0]
        Y = Y.view(-1)
        # X = [-1.97, -1.70, -1.91, -2.16]
        X = X[range(X.shape[0]), Y]
        # mask = [1., 1., 1., 0.] (for <PAD>)
        mask = (Y > 0).float()
        # X = [-1.97, -1.70, -1.91, -0.00]
        X = X * mask
        # x_num = 3
        x_num = int(torch.sum(mask))
        # -1 * (-1.97 + -1.70 + -1.91 + -0.00) / 3
        return -torch.sum(X) / x_num

    def _extend(self, results, Y_hat, mask, indices):
        _, __results = Y_hat.max(-1)
        __results = __results * mask
        _results = [None] * len(Y_hat)
        for i, index in enumerate(indices):
            _results[index] = [y for y in __results[i].tolist() if y != 0]
        results.extend(_results)
