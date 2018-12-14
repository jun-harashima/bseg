import torch
from model import Model
from dataset import Dataset


EMBEDDING_DIM = 2
HIDDEN_DIM = 4

examples = [
    (("人参", "を", "切る"), ("名詞", "助詞", "動詞")),
    (("ざっくり", "切る"), ("副詞", "動詞")),
    (("葱", "は", "細く", "刻む"), ("名詞", "助詞", "形容詞", "動詞"))
]
dataset = Dataset(examples)

model = Model(EMBEDDING_DIM, HIDDEN_DIM, dataset.word_to_index,
              dataset.tag_to_index, 3)
model.train(dataset)
torch.save(model.state_dict(), 'pos.model')

model.load_state_dict(torch.load('pos.model'))
examples = [
    (("葱", "を", "切る"), ("名詞", "助詞", "動詞")),
    (("細く", "切る"), ("副詞", "動詞")),
    (("大根", "は", "ざっくり", "刻む"), ("名詞", "助詞", "形容詞", "動詞"))
]
dataset = Dataset(examples)
results = model.test(dataset)
print(results)
