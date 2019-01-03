class OneInputDataset():

    def __init__(self, examples, y_to_index=None, x_to_index=None):
        tags_set = [example['Y'] for example in examples]
        self.y_to_index = y_to_index
        if y_to_index is None:
            tags = [tag for tags in tags_set for tag in tags]
            self.y_to_index = self._make_index(tags)
        self.Y = self._degitize(tags_set, self.y_to_index)

        self.x_to_index = x_to_index
        if x_to_index is None:
            self.x_to_index = []
            for i in range(len(examples[0]['Xs'])):
                tokens_set = [example['Xs'][i] for example in examples]
                tokens = [token for tokens in tokens_set for token in tokens]
                self.x_to_index.append(self._make_index(tokens))

        self.Xs = []
        for i in range(len(examples[0]['Xs'])):
            tokens_set = [example['Xs'][i] for example in examples]
            self.Xs.append(self._degitize(tokens_set, self.x_to_index[i]))

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
