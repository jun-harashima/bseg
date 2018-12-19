import torch
import torch.nn as nn
from bseg.model import Model


torch.manual_seed(1)


class WithPos(Model):

    def __init__(self, embedding_dim, hidden_dim, E_pos, H_pos, word_to_index,
                 pos_to_index, tag_to_index, word_pad_index=0, pos_pad_index=0,
                 tag_pad_index=0, batch_size=16):
        super(WithPos, self).__init__()
        self.E_pos = E_pos  # embedding dimension for POS
        self.H_pos = H_pos  # hidden dimension for POS
        self.P = len(pos_to_index)
        self.pos_embeddings = self._init_pos_embeddings()
        self.lstm = self._init_lstm()
        self.hidden2tag = self._init_hidden2tag()

    def _init_pos_embeddings(self):
        embeddings = nn.Embedding(self.P, self.E_pos, self.pos_pad_index)
        return embeddings.cuda() if torch.cuda.is_available() else embeddings

    def _init_lstm(self):
        lstm = nn.LSTM(self.embedding_dim + self.E_pos,
                       self.hidden_dim + self.H_pos, bidirectional=True)
        return lstm.cuda() if torch.cuda.is_available() else lstm

    def _init_hidden2tag(self):
        hidden2tag = nn.Linear((self.hidden_dim + self.H_pos) * 2,
                               self.tagset_size)
        return hidden2tag.cuda() if torch.cuda.is_available() else hidden2tag

    def _init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        zeros = torch.zeros(2, self.batch_size, (self.hidden_dim + self.H_pos),
                            device=self.device)
        return (zeros, zeros)
