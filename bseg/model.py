import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


torch.manual_seed(1)


class Model(nn.Module):

    def __init__(self, dataset, embedding_dim, hidden_dim):
        super(Model, self).__init__()
        self.dataset = dataset
        self.hidden_dim = hidden_dim

        self.word_embeddings = \
            nn.Embedding(len(dataset.word_to_index), embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, len(dataset.tag_to_index))
        self.hidden = self._initialize_hidden()

    def _initialize_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

    def train(self):
        loss_function = nn.NLLLoss()
        optimizer = optim.SGD(self.parameters(), lr=0.1)
        for epoch in range(10):
            for (words, tags) in self.dataset.degitized_examples:
                # Step 1. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                self.zero_grad()

                # Also, we need to clear out the hidden state of the LSTM,
                # detaching it from its history on the last instance.
                self.hidden = self._initialize_hidden()

                # Step 2. Run our forward pass.
                scores = self(words)

                # Step 3. Compute the loss, gradients, and update the
                # parameters by calling optimizer.step()
                loss = loss_function(scores, tags)
                loss.backward()
                optimizer.step()
