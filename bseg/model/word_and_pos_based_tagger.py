import torch
from bseg.model.word_based_tagger import WordBasedTagger


torch.manual_seed(1)


class WordAndPosBasedTagger(WordBasedTagger):

    def _train(self, optimizer, batches, epoch, dev_set):
        loss_sum = 0
        for Y, X, X2 in batches:
            self.zero_grad()
            self.hidden = self._init_hidden()
            X, lengths, _ = self._tensorize(X)
            X2, lengths, _ = self._tensorize(X2)
            Y, lengths, _ = self._tensorize(Y)
            Y_hat = self([X, X2], lengths)
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
        for _, X, X2 in batches:
            self.hidden = self._init_hidden()
            X, lengths, indices = self._tensorize(X)
            X2, _, _ = self._tensorize(X2)
            mask = (X > 0).long()
            Y_hat = self([X, X2], lengths)
            self._extend(results, Y_hat, mask, indices)
        return results
