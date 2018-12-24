from bseg.dataset.base_dataset import BaseDataset


class TwoInputDataset(BaseDataset):

    def __init__(self, examples, word_to_index=None, pos_to_index=None,
                 tag_to_index=None):
        self.word_to_index = word_to_index
        self.pos_to_index = pos_to_index
        self.tag_to_index = tag_to_index
        if not all([word_to_index, pos_to_index, tag_to_index]):
            self.word_to_index, self.pos_to_index, self.tag_to_index = \
                self._make_index(examples)
        self.X1, self.X2, self.Y = self._degitize(examples)

    def _make_index(self, examples):
        word_to_index = {'<PAD>': 0, '<UNK>': 1}
        pos_to_index = {'<PAD>': 0, '<UNK>': 1}
        tag_to_index = {'<PAD>': 0, '<UNK>': 1}
        for (words, poses, tags) in examples:
            for word in words:
                self._add_index(word, word_to_index)
            for pos in poses:
                self._add_index(pos, pos_to_index)
            for tag in tags:
                self._add_index(tag, tag_to_index)
        return word_to_index, pos_to_index, tag_to_index

    def _degitize(self, examples):
        X1 = []
        X2 = []
        Y = []
        for (words, poses, tags) in examples:
            _X1 = [self._get_index(word, self.word_to_index) for word in words]
            _X2 = [self._get_index(pos, self.pos_to_index) for pos in poses]
            _Y = [self._get_index(tag, self.tag_to_index) for tag in tags]
            X1.append(_X1)
            X2.append(_X2)
            Y.append(_Y)
        return X1, X2, Y
