import torch
from model import Model
from dataset import Dataset


EMBEDDING_DIM = 6
HIDDEN_DIM = 6

examples = [(("人参", "を", "切る"), ("名詞", "助詞", "動詞")),
            (("大根", "を", "切る"), ("名詞", "助詞", "動詞")),
            (("白菜", "は", "蒸す"), ("名詞", "助詞", "動詞")),
            (("牛蒡", "は", "削ぐ"), ("名詞", "助詞", "動詞"))]
dataset = Dataset(examples)

model = Model(EMBEDDING_DIM, HIDDEN_DIM, len(dataset.word_to_index),
              len(dataset.tag_to_index))
model.train(dataset)
