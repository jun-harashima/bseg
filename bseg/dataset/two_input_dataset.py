from bseg.dataset import Dataset


class TwoInputDataset(Dataset):

    def __init__(self, examples, x1_to_index=None, x2_to_index=None,
                 y_to_index=None):
        self.x1_to_index, self.x2_to_index, self.y_to_index = \
            x1_to_index, x2_to_index, y_to_index
        if not all([x1_to_index, x2_to_index, y_to_index]):
            self.x1_to_index, self.x2_to_index, self.y_to_index = \
                self._make_index(examples)
        self.X1, self.X2, self.Y = self._degitize(examples)

    def _make_index(self, examples):
        x1_to_index = {"<PAD>": 0, "<UNK>": 1}
        x2_to_index = {"<PAD>": 0, "<UNK>": 1}
        y_to_index = {"<PAD>": 0, "<UNK>": 1}
        for (_X1, _X2, _Y) in examples:
            for x1 in _X1:
                self._add_index(x1, x1_to_index)
            for x2 in _X2:
                self._add_index(x2, x2_to_index)
            for y in _Y:
                self._add_index(y, y_to_index)
        return x1_to_index, x2_to_index, y_to_index

    def _degitize(self, examples):
        X1 = []
        X2 = []
        Y = []
        for (_X1, _X2, _Y) in examples:
            X1.append([self._get_index(x1, self.x1_to_index) for x1 in _X1])
            X2.append([self._get_index(x2, self.x2_to_index) for x2 in _X2])
            Y.append([self._get_index(y, self.y_to_index) for y in _Y])
        return X1, X2, Y
