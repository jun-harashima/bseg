import torch
from bseg.with_pos import WithPos
from bseg.dataset.two_input_dataset import TwoInputDataset


EMBEDDING_DIM = 2
HIDDEN_DIM = 4
E_pos = 2
H_pos = 4

examples = [
    (('B-S', 'B-I', 'B-P'), ('人参', 'を', '切る'), ('名詞', '助詞', '動詞')),
    (('B-M', 'B-P'), ('ざっくり', '切る'), ('副詞', '動詞')),
    (('B-S', 'B-I', 'B-M', 'B-P'), ('葱', 'は', '細く', '刻む'),
     ('名詞', '助詞', '形容詞', '動詞'))
]
dataset = TwoInputDataset(examples)

model = WithPos(EMBEDDING_DIM, HIDDEN_DIM, E_pos, H_pos, dataset.tag_to_index,
                dataset.word_to_index, dataset.pos_to_index, batch_size=3)
model.train(dataset)
torch.save(model.state_dict(), 'with_pos.model')

model.load_state_dict(torch.load('with_pos.model'))
examples = [
    (('B-S', 'B-I', 'B-P'), ('葱', 'を', '切る'), ('名詞', '助詞', '動詞')),
    (('B-M', 'B-P'), ('細く', '切る'), ('副詞', '動詞')),
    (('B-S', 'B-I', 'B-M', 'B-P'), ('大根', 'は', 'ざっくり', '刻む'),
     ('名詞', '助詞', '形容詞', '動詞'))
]
dataset = TwoInputDataset(examples)
results = model.test(dataset)
print(results)
