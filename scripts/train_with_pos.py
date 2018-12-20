import torch
from model import Model
from with_pos import WithPos
from dataset import Dataset


EMBEDDING_DIM = 2
HIDDEN_DIM = 4

examples = [
    (("人参", "を", "切る"), ("名詞", "助詞", "動詞"), ("B-S", "B-I", "B-P")),
    (("ざっくり", "切る"), ("副詞", "動詞"), ("B-M", "B-P")),
    (("葱", "は", "細く", "刻む"), ("名詞", "助詞", "形容詞", "動詞"),
     ("B-S", "B-I", "B-M", "B-P"))
]
dataset = Dataset(examples)

model = WithPos(EMBEDDING_DIM, HIDDEN_DIM, dataset.word_to_index,
                dataset.tag_to_index, batch_size=3)
model.train(dataset)
torch.save(model.state_dict(), 'with_pos.model')

model.load_state_dict(torch.load('wit_pos.model'))
examples = [
    (("葱", "を", "切る"), ("名詞", "助詞", "動詞"), ("B-S", "B-I", "B-P")),
    (("細く", "切る"), ("副詞", "動詞"), ("B-M", "B-P")),
    (("大根", "は", "ざっくり", "刻む"), ("名詞", "助詞", "形容詞", "動詞"),
     ("B-S", "B-I", "B-M", "B-P"))
]
dataset = Dataset(examples)
results = model.test(dataset)
print(results)
