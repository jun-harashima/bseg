import torch
from model import Model
from pos_tagging_dataset import PosTaggingDataset
from torch.utils.data import DataLoader


EMBEDDING_DIM = 6
HIDDEN_DIM = 6

examples = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
dataset = PosTaggingDataset(examples)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

model = Model(EMBEDDING_DIM, HIDDEN_DIM, len(loader.dataset.word_to_index),
              len(loader.dataset.tag_to_index), loader.batch_size)
model.train(loader)
