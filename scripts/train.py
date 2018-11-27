import torch
import torch.optim as optim
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

model = Model(len(word_to_index), tag_to_index, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

# Check predictions before training
with torch.no_grad():
    precheck_sent = model.prepare_sequence_for(training_data[0][0], word_to_index)
    precheck_tags = torch.tensor([tag_to_index[t] for t in training_data[0][1]], dtype=torch.long)
    print(model(precheck_sent))

# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(
        300):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        sentence_in = model.prepare_sequence_for(sentence, word_to_index)
        targets = torch.tensor([tag_to_index[t] for t in tags], dtype=torch.long)

        # Step 3. Run our forward pass.
        loss = model.neg_log_likelihood(sentence_in, targets)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()

# Check predictions after training
with torch.no_grad():
    precheck_sent = model.prepare_sequence_for(training_data[0][0], word_to_index)
    print(model(precheck_sent))
# We got it!
