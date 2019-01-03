import torch
from bseg.model import Model
from bseg.dataset import Dataset


EMBEDDING_DIMS = [2]
HIDDEN_DIMS = [4]

examples = [
    {'Xs': [['人参', 'を', '切る']],
     'Y': ['名詞', '助詞', '動詞']},
    {'Xs': [['ざっくり', '切る']],
     'Y': ['副詞', '動詞']},
    {'Xs': [['葱', 'は', '細く', '刻む']],
     'Y': ['名詞', '助詞', '形容詞', '動詞']}
]
dataset = Dataset(examples)

x_set_sizes = [len(dataset.x_to_index[0])]
y_set_size = len(dataset.y_to_index)
model = Model(EMBEDDING_DIMS, HIDDEN_DIMS, x_set_sizes, y_set_size,
              batch_size=3)
model.train(dataset)
torch.save(model.state_dict(), 'one_input.model')

model.load_state_dict(torch.load('one_input.model'))
examples = [
    {'Xs': [['葱', 'を', '切る']],
     'Y': ['名詞', '助詞', '動詞']},
    {'Xs': [['細く', '切る']],
     'Y': ['副詞', '動詞']},
    {'Xs': [['大根', 'は', 'ざっくり', '刻む']],
     'Y': ['名詞', '助詞', '形容詞', '動詞']}
]
dataset = Dataset(examples)
results = model.test(dataset)
print(results)
