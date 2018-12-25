class OneInputDataset():

    def __init__(self, examples, tag_to_index=None, word_to_index=None):
        self.tag_to_index = tag_to_index
        if tag_to_index is None:
            tags = [tag for example in examples for tag in example[0]]
            self.tag_to_index = self._make_index(tags)

        self.word_to_index = word_to_index
        if word_to_index is None:
            words = [word for example in examples for word in example[1]]
            self.word_to_index = self._make_index(words)

        self.X, self.Y = self._degitize(examples)

    def _make_index(self, tokens):
        token_to_index = {'<PAD>': 0, '<UNK>': 1}
        for token in tokens:
            if token not in token_to_index:
                token_to_index[token] = len(token_to_index)
        return token_to_index

    def _get_index(self, token, token_to_index):
        if token not in token_to_index:
            return token_to_index['<UNK>']
        return token_to_index[token]

    def _degitize(self, examples):
        Y = []
        X = []
        for (tags, words) in examples:
            _Y = [self._get_index(tag, self.tag_to_index) for tag in tags]
            _X = [self._get_index(word, self.word_to_index) for word in words]
            Y.append(_Y)
            X.append(_X)
        return Y, X
