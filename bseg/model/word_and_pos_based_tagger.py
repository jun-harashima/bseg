import torch
import torch.nn as nn
from bseg.model.word_based_tagger import WordBasedTagger


torch.manual_seed(1)


class WordAndPosBasedTagger(WordBasedTagger):

    def __init__(self, embedding_dim, hidden_dim, pos_embedding_dim,
                 pos_hidden_dim, tag_to_index, word_to_index, pos_to_index,
                 tag_pad_index=0, word_pad_index=0, pos_pad_index=0,
                 batch_size=16):
        self.pos_embedding_dim = pos_embedding_dim
        self.pos_hidden_dim = pos_hidden_dim
        self.posset_size = len(pos_to_index)
        self.pos_pad_index = word_pad_index
        super(WordAndPosBasedTagger, self).__init__(embedding_dim, hidden_dim,
                                                    word_to_index,
                                                    tag_to_index)
        self.pos_embeddings = self._init_pos_embeddings()

    def _init_pos_embeddings(self):
        pos_embeddings = nn.Embedding(self.posset_size, self.pos_embedding_dim,
                                      self.pos_pad_index)
        return pos_embeddings.cuda() if self.use_cuda else pos_embeddings

    def _init_lstm(self):
        lstm = nn.LSTM(self.embedding_dim + self.pos_embedding_dim,
                       self.hidden_dim + self.pos_hidden_dim,
                       bidirectional=True)
        return lstm.cuda() if self.use_cuda else lstm

    def _init_hidden2tag(self):
        hidden2tag = nn.Linear((self.hidden_dim + self.pos_hidden_dim) * 2,
                               self.tagset_size)
        return hidden2tag.cuda() if self.use_cuda else hidden2tag

    def _init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        zeros = torch.zeros(2, self.batch_size,
                            (self.hidden_dim + self.pos_hidden_dim),
                            device=self.device)
        return (zeros, zeros)

    def forward(self, X, X2, lengths):
        X = self._embed(X)
        X2 = self._pos_embed(X2)
        X = torch.cat((X, X2), 2)
        X = self._forward(X, lengths)
        return X

    def _train(self, optimizer, batches, epoch, dev_set):
        loss_sum = 0
        for X, X2, Y in batches:
            self.zero_grad()
            self.hidden = self._init_hidden()
            X, lengths, _ = self._tensorize(X, self.word_pad_index)
            X2, lengths, _ = self._tensorize(X2, self.pos_pad_index)
            Y, lengths, _ = self._tensorize(Y, self.tag_pad_index)
            Y_hat = self(X, X2, lengths)
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
        for X, X2, _ in batches:
            self.hidden = self._init_hidden()
            X, lengths, indices = self._tensorize(X, self.word_pad_index)
            X2, _, _ = self._tensorize(X2, self.pos_pad_index)
            mask = (X > 0).long()
            Y_hat = self(X, X2, lengths)
            self._extend(results, Y_hat, mask, indices)
        return results

    def _split(self, dataset):
        if len(dataset.X) < self.batch_size:
            self.batch_size = len(dataset.X)
        return list(zip(zip(*[iter(dataset.X)]*self.batch_size),
                        zip(*[iter(dataset.X2)]*self.batch_size),
                        zip(*[iter(dataset.Y)]*self.batch_size)))

    def _pos_embed(self, X2):
        return self.pos_embeddings(X2)