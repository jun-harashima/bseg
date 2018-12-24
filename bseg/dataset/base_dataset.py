class BaseDataset():

    def _add_index(self, token, token_to_index):
        if token not in token_to_index:
            token_to_index[token] = len(token_to_index)

    def _get_index(self, token, token_to_index):
        if token not in token_to_index:
            return token_to_index['<UNK>']
        return token_to_index[token]
