import torch
from bseg.model import Model
from bseg.dataset.one_input_dataset import OneInputDataset


EMBEDDING_DIMS = [2, 2]
HIDDEN_DIMS = [4, 4]

examples = [
    {'Xs': [['人参', 'を', '切る'],
            ['名詞', '助詞', '動詞']],
     'Y': ['B-S', 'B-I', 'B-P']},
    {'Xs': [['ざっくり', '切る'],
            ['副詞', '動詞']],
     'Y': ['B-M', 'B-P']},
    {'Xs': [['葱', 'は', '細く', '刻む'],
            ['名詞', '助詞', '形容詞', '動詞']],
     'Y': ['B-S', 'B-I', 'B-M', 'B-P']}
]
dataset = OneInputDataset(examples)

tag_num = len(dataset.y_to_index)
token_nums = [len(dataset.x_to_index[0]), len(dataset.x_to_index[1])]
model = Model(EMBEDDING_DIMS, HIDDEN_DIMS, tag_num, token_nums, batch_size=3)
model.train(dataset)
torch.save(model.state_dict(), 'word_and_pos_based_tagger.model')

model.load_state_dict(torch.load('word_and_pos_based_tagger.model'))
examples = [
    {'Xs': [['葱', 'を', '切る'],
            ['名詞', '助詞', '動詞']],
     'Y': ['B-S', 'B-I', 'B-P']},
    {'Xs': [['細く', '切る'],
            ['副詞', '動詞']],
     'Y': ['B-M', 'B-P']},
    {'Xs': [['大根', 'は', 'ざっくり', '刻む'],
            ['名詞', '助詞', '形容詞', '動詞']],
     'Y': ['B-S', 'B-I', 'B-M', 'B-P']}
]
dataset = OneInputDataset(examples)
results = model.test(dataset)
print(results)
