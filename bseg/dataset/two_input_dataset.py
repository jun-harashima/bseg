from bseg.dataset.one_input_dataset import OneInputDataset


class TwoInputDataset(OneInputDataset):

    def __init__(self, examples, y_to_index=None, x_to_index=None):
        super(TwoInputDataset, self).__init__(examples, y_to_index, x_to_index)
        poses_set = [example['Xs'][1] for example in examples]
        poses = [pos for poses in poses_set for pos in poses]
        self.x_to_index.append(self._make_index(poses))
        self.Xs.append(self._degitize(poses_set, self.x_to_index[1]))
