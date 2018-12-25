from bseg.dataset.one_input_dataset import OneInputDataset


class TwoInputDataset(OneInputDataset):

    def __init__(self, examples, tag_to_index=None, word_to_index=None,
                 pos_to_index=None):
        super(OneInputDataset, self).__init__(examples, tag_to_index=None,
                                              word_to_index=None)
        self.pos_to_index = pos_to_index
        if pos_to_index is None:
            poses = [pos for example in examples for pos in example[2]]
            self.pos_to_index = self._make_index(poses)

        self.X1, self.X2, self.Y = self._degitize(examples)

    def _degitize(self, examples):
        Y = []
        X1 = []
        X2 = []
        for (tags, words, poses) in examples:
            _Y = [self._get_index(tag, self.tag_to_index) for tag in tags]
            _X1 = [self._get_index(word, self.word_to_index) for word in words]
            _X2 = [self._get_index(pos, self.pos_to_index) for pos in poses]
            Y.append(_Y)
            X1.append(_X1)
            X2.append(_X2)
        return Y, X1, X2
