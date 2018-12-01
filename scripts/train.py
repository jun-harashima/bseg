import torch
from util import Util
from model import Model


EMBEDDING_DIM = 5
HIDDEN_DIM = 4

# Make up some training data
raw_training_data = [(
    "the wall street journal reported today that apple corporation made money".split(),
    "B I I I O O O B I O O".split()
), (
    "georgia tech is a university in georgia".split(),
    "B I O O O O B".split()
)]

util = Util()
word_to_index = util.make_index_from(raw_training_data)
tag_to_index = {"BOS": 0, "EOS": 1, "B": 2, "I": 3, "O": 4}

training_data = []
for words, tags in raw_training_data:
    word_indexes = util.degitize(words, word_to_index)
    tag_indexes = util.degitize(tags, tag_to_index)
    training_data.append((word_indexes, tag_indexes))

model = Model(word_to_index, tag_to_index, EMBEDDING_DIM, HIDDEN_DIM)
model.train(training_data)
