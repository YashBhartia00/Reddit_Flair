import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import math

class Encoder(nn.Module):
  def __init__(self, embedding_dim, hidden_dim, nlayers=1, dropout=0.,
               bidirectional=True):
    super(Encoder, self).__init__()
    """
    Arguments
    ----------
    embedding_dim : Embedding dimention of GloVe embeddings
    hidden_dim : size of hidden layer of LSTM
    nlayers : number of hidden layers of LSTM
    dropout : 
    bidirectional : True if using bidirectional LSTM

    """

    self.bidirectional = bidirectional
    self.rnn = nn.LSTM(embedding_dim, hidden_dim, nlayers, 
                        dropout=dropout, bidirectional=bidirectional)

  def forward(self, input, hidden=None):
    """
    Parameters
    ---------
    input:  embeddings of the sentence
    hidden: tensor contatain initial hidden layers and cell states

    Returns
    ---------
    Output layer, hidden layer and cell state
    """
    return self.rnn(input, hidden)

class Attention(nn.Module):
  def __init__(self, query_dim, key_dim, value_dim):
    super(Attention, self).__init__()
    """
    Arguments
    ---------
    query_dim :
    key_dim :
    value_dim: 

    """
    self.scale = 1. / math.sqrt(query_dim)

  def forward(self, query, keys, values):
    """
    Parameters
    ---------
    input:  embeddings of the sentence
    hidden: tensor contatain initial hidden layers and cell states

    Returns
    ---------
    Output layer, hidden layer and cell state
    """
    query = query.unsqueeze(1)
    keys = keys.transpose(0,1).transpose(1,2)
    bmm = torch.bmm(query, keys) 
    final = F.softmax(bmm.mul_(self.scale), dim=2) 

    values = values.transpose(0,1)
    linear_combination = torch.bmm(final, values).squeeze(1) 
    return final, linear_combination

class Classifier(nn.Module):
  def __init__(self, embedding, encoder, attention, hidden_dim, num_classes, baseline=False):
    super(Classifier, self).__init__()
    self.embedding = embedding
    self.encoder = encoder
    self.attention = attention
    self.decoder = nn.Linear(hidden_dim, num_classes)
    self.baseline = baseline

  def forward(self, input):
    outputs, hidden = self.encoder(self.embedding(input))
    if isinstance(hidden, tuple):
      hidden = hidden[1]

    if self.encoder.bidirectional:
      hidden = torch.cat([hidden[-1], hidden[-2]], dim=1)
    else:
      hidden = hidden[-1]

    final_output, linear_combination = self.attention(hidden, outputs, outputs) 
    logits = self.decoder(linear_combination) if not self.baseline else self.decoder(hidden)

    return logits, final_output
