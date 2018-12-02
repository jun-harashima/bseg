import torch
from model import Model
from pos_tagging_dataset import PosTaggingDataset


EMBEDDING_DIM = 6
HIDDEN_DIM = 6

examples = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
dataset = PosTaggingDataset(examples)

model = Model(EMBEDDING_DIM, HIDDEN_DIM, len(dataset.word_to_index),
              len(dataset.tag_to_index))
model.train(dataset)
