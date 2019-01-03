# z: a token (e.g., word, pos, and tag).
# Z: a sequence of tokens.
# Z_set: a set of sequences.
# zs: a concatnation of sets.


class Dataset():

    def __init__(self, examples, y_to_index=None, x_to_index=None):
        Y_set = [example['Y'] for example in examples]
        self.y_to_index = y_to_index
        if y_to_index is None:
            ys = [y for Y in Y_set for y in Y]
            self.y_to_index = self._make_index(ys)
        self.Y = self._degitize(Y_set, self.y_to_index)

        self.x_to_index = x_to_index
        if x_to_index is None:
            self.x_to_index = []
            for i in range(len(examples[0]['Xs'])):
                X_set = [example['Xs'][i] for example in examples]
                xs = [x for X in X_set for x in X]
                self.x_to_index.append(self._make_index(xs))

        self.Xs = []
        for i in range(len(examples[0]['Xs'])):
            X_set = [example['Xs'][i] for example in examples]
            self.Xs.append(self._degitize(X_set, self.x_to_index[i]))

    def _make_index(self, zs):
        z_to_index = {'<PAD>': 0, '<UNK>': 1}
        for z in zs:
            if z not in z_to_index:
                z_to_index[z] = len(z_to_index)
        return z_to_index

    def _get_index(self, z, z_to_index):
        if z not in z_to_index:
            return z_to_index['<UNK>']
        return z_to_index[z]

    def _degitize(self, Z_set, z_to_index):
        Z = []
        for _Z in Z_set:
            _Z = [self._get_index(z, z_to_index) for z in _Z]
            Z.append(_Z)
        return Z
