import torch


class Util:

    def __init__(self):
        pass

    def make_index_from(self, training_data):
        word_to_index = {}
        for sentence, tags in training_data:
            for word in sentence:
                if word not in word_to_index:
                    word_to_index[word] = len(word_to_index)
        return word_to_index

    def degitize(self, tokens, token_to_index):
        indexes = [token_to_index[token] for token in tokens]
        return torch.tensor(indexes, dtype=torch.long)
