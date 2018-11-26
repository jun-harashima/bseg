import torch
from util import Util
from model import Model


EMBEDDING_DIM = 5
HIDDEN_DIM = 4

# Make up some training data
training_data = [(
    "the wall street journal reported today that apple corporation made money".split(),
    "B I I I O O O B I O O".split()
), (
    "georgia tech is a university in georgia".split(),
    "B I O O O O B".split()
)]

util = Util()
word_to_index = util.make_index_from(training_data)
tag_to_index = {"B": 0, "I": 1, "O": 2, "<START>": 3, "<STOP>": 4}

model = Model(word_to_index, tag_to_index, EMBEDDING_DIM, HIDDEN_DIM)
model.train(training_data)

# Check predictions after training
with torch.no_grad():
    precheck_sent = model.degitize(training_data[0][0], word_to_index)
    print(model(precheck_sent))
