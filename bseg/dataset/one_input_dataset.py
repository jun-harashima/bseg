from bseg.dataset.base_dataset import BaseDataset


class OneInputDataset(BaseDataset):

    def __init__(self, examples, word_to_index=None, tag_to_index=None):
        self.word_to_index, self.tag_to_index = word_to_index, tag_to_index
        if word_to_index is None or tag_to_index is None:
            self.word_to_index, self.tag_to_index = self._make_index(examples)
        self.X, self.Y = self._degitize(examples)

    def _make_index(self, examples):
        word_to_index = {"<PAD>": 0, "<UNK>": 1}
        tag_to_index = {"<PAD>": 0, "<UNK>": 1}
        for (words, tags) in examples:
            for word in words:
                self._add_index(word, word_to_index)
            for tag in tags:
                self._add_index(tag, tag_to_index)
        return word_to_index, tag_to_index

    def _degitize(self, examples):
        X = []
        Y = []
        for (words, tags) in examples:
            _X = [self._get_index(word, self.word_to_index) for word in words]
            _Y = [self._get_index(tag, self.tag_to_index) for tag in tags]
            X.append(_X)
            Y.append(_Y)
        return X, Y
