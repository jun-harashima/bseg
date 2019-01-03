# z: a token (e.g., word, pos, and tag).
# Z: a sequence of tokens.
# Z_set: a set of sequences.
# zs: a concatnation of sets.


class Dataset():

    def __init__(self, examples, x_to_index=None, y_to_index=None):
        self.x_to_index = x_to_index
        X_sets = [[example['Xs'][i] for example in examples]
                  for i in range(len(examples[0]['Xs']))]
        if x_to_index is None:
            self.x_to_index = []
            for i in range(len(examples[0]['Xs'])):
                xs = [x for X in X_sets[i] for x in X]
                self.x_to_index.append(self._make_index(xs))

        self.y_to_index = y_to_index
        Y_set = [example['Y'] for example in examples]
        if y_to_index is None:
            ys = [y for Y in Y_set for y in Y]
            self.y_to_index = self._make_index(ys)

        self.Xs = []
        for i in range(len(examples[0]['Xs'])):
            self.Xs.append(self._degitize(X_sets[i], self.x_to_index[i]))

        self.Y = self._degitize(Y_set, self.y_to_index)

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
