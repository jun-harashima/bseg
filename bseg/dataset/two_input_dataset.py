from bseg.dataset.one_input_dataset import OneInputDataset


class TwoInputDataset(OneInputDataset):

    def __init__(self, examples, y_to_index=None, word_to_index=None,
                 pos_to_index=None):
        super(TwoInputDataset, self).__init__(examples, y_to_index,
                                              word_to_index)

        poses_set = [example['Xs'][1] for example in examples]
        self.pos_to_index = pos_to_index
        if pos_to_index is None:
            poses = [pos for poses in poses_set for pos in poses]
            self.pos_to_index = self._make_index(poses)
        self.Xs.append(self._degitize(poses_set, self.pos_to_index))
