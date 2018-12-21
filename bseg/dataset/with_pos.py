from bseg.dataset import Dataset


class WithPos(Dataset):

    def __init__(self, examples, word_to_index=None, pos_to_index=None,
                 tag_to_index=None):
        super(WithPos, self).__init__()
        self.word_to_index, self.pos_to_index, self.tag_to_index = \
            word_to_index, pos_to_index, tag_to_index
        if all([word_to_index, pos_to_index, tag_to_index]):
            self.word_to_index, self.pos_to_index, self.tag_to_index = \
                self._make_index(examples)
        self.W, self.P, self.T = self._degitize(examples)

    def _make_index(self, examples):
        word_to_index = {"<PAD>": 0, "<UNK>": 1}
        pos_to_index = {"<PAD>": 0, "<UNK>": 1}
        tag_to_index = {"<PAD>": 0, "<UNK>": 1}
        for (words, poses, tags) in examples:
            for word in words:
                self._add_index(word, word_to_index)
            for pos in poses:
                self._add_index(pos, pos_to_index)
            for tag in tags:
                self._add_index(tag, tag_to_index)
        return word_to_index, pos_to_index, tag_to_index

    def _degitize(self, examples):
        W = []  # Words
        P = []  # POSes
        T = []  # Tags
        for (words, poses, tags) in examples:
            _W = [self._get_index(word, self.word_to_index) for word in words]
            _P = [self._get_index(pos, self.pos_to_index) for pos in poses]
            _T = [self._get_index(tag, self.tag_to_index) for tag in tags]
            W.append(_W)
            P.append(_P)
            T.append(_T)
        return W, P, T
