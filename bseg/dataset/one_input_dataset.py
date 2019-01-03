class OneInputDataset():

    def __init__(self, examples, tag_to_index=None, word_to_index=None):
        tags_set = [example['Y'] for example in examples]
        self.tag_to_index = tag_to_index
        if tag_to_index is None:
            tags = [tag for tags in tags_set for tag in tags]
            self.tag_to_index = self._make_index(tags)
        self.Y = self._degitize(tags_set, self.tag_to_index)

        words_set = [example['Xs'][0] for example in examples]
        self.word_to_index = word_to_index
        if word_to_index is None:
            words = [word for words in words_set for word in words]
            self.word_to_index = self._make_index(words)
        self.Xs = [self._degitize(words_set, self.word_to_index)]

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

    def _degitize(self, tokens_set, token_to_index):
        Z = []
        for tokens in tokens_set:
            _Z = [self._get_index(token, token_to_index) for token in tokens]
            Z.append(_Z)
        return Z
