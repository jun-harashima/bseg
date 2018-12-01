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

model = Model(dataset, EMBEDDING_DIM, HIDDEN_DIM)
model.train()

with torch.no_grad():
    word_indexes = dataset._degitize(examples[0][0], dataset.word_to_index)
    scores = model(word_indexes)
    # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
    # for word i. The predicted tag is the maximum scoring tag.
    # Here, we can see the predicted sequence below is 0 1 2 0 1
    # since 0 is index of the maximum value of row 1,
    # 1 is the index of maximum value of row 2, etc.
    # Which is DET NOUN VERB DET NOUN, the correct sequence!
    print(scores)
