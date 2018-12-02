import torch
from torch.utils.data import Dataset


class PosTaggingDataset(Dataset):

    def __init__(self, examples):
        self.word_to_index, self.tag_to_index = self._make_index(examples)
        self.degitized_examples = self._degitize_all(examples)

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

    def _degitize_all(self, examples):
        degitized_examples = []
        for (words, tags) in examples:
            word_indexes = self._degitize(words, self.word_to_index)
            tag_indexes = self._degitize(tags, self.tag_to_index)
            degitized_examples.append((word_indexes, tag_indexes))
        return degitized_examples

    def _degitize(self, tokens, token_to_index):
        indexes = [token_to_index[token] for token in tokens]
        return torch.tensor(indexes, dtype=torch.long)
