from torch.utils.data import Dataset


class PosTaggingDataset(Dataset):

    def __init__(self, examples):
        self.word_to_index, self.tag_to_index = self._make_index(examples)
        self.X, self.Y = self._degitize(examples)

    def __len__(self):
        return len(self.degitized_examples)

    def __getitem__(self, i):
        return self.degitized_examples[i]

    def _make_index(self, examples):
        word_to_index = {"PAD": 0}
        tag_to_index = {"PAD": 0}
        for (words, tags) in examples:
            for word in words:
                self._add_index(word, word_to_index)
            for tag in tags:
                self._add_index(tag, tag_to_index)
        return word_to_index, tag_to_index

    def _add_index(self, token, token_to_index):
        if token not in token_to_index:
            token_to_index[token] = len(token_to_index)

    def _degitize(self, examples):
        X = []
        Y = []
        for (words, tags) in examples:
            X.append([self.word_to_index[word] for word in words])
            Y.append([self.tag_to_index[tag] for tag in tags])
        return X, Y
