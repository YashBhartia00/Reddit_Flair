from model import Encoder, Attention, Classifier
from data import dataload
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from torchtext import data

def train(epoch, model, trainiter, valiter, optimizer, criterion):
  model.train()
  train_accuracy = [0,0]
  train_total_loss = 0
  for batch_num, batch in enumerate(trainiter):
    model.zero_grad()
    x, lens = batch.text
    y = batch.label
    
    logits, _ = model(x.to(device))
    loss = criterion(logits, y)
    train_total_loss += float(loss)
    train_accuracy = [train_accuracy[i] + [f1_score(y.cpu(),torch.max(logits, 1)[1].cpu(),  average='macro'),
                                       f1_score(y.cpu(),torch.max(logits, 1)[1].cpu(), average='micro')][i] for i in range(2)]
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
    optimizer.step()

  model.eval()
  val_accuracy = [0,0]
  val_total_loss = 0
  with torch.no_grad():
    for batch_num, batch in enumerate(valiter):
      x, lens = batch.text
      y = batch.label

      logits, _ = model(x)
      val_total_loss += float(criterion(logits, y))
      val_accuracy = [val_accuracy[i] + [f1_score(y.cpu(), torch.max(logits, 1)[1].cpu(), average='macro'),
                                     f1_score(y.cpu(), torch.max(logits, 1)[1].cpu(),  average='micro')][i] for i in range(2)]

  train_lo = (train_total_loss / len(trainiter))
  train_acc = [train_accuracy[i] / len(trainiter) for i in range(2)]
  val_lo = (val_total_loss / len(valiter))
  val_acc = [val_accuracy[i] / len(valiter) for i in range(2)]

  print(f'Epoch {epoch}: train_loss: {train_lo:.4f} f1_macro: {train_acc[0]}  f1_micro {train_acc[1]} | val_loss: {val_lo:.4f} f1_macro: {val_acc[0]}  f1_micro {val_acc[1]}')
  return model


if __name__ == '__main__':

    cuda = torch.cuda.is_available()
    device = torch.device("cpu") if not cuda else torch.device("cuda:0")
    dataL = dataload()

    text_field, label_field, trainds, valds, vectors = dataL.buildVocab()
    values = [label_field.vocab.freqs.most_common(15)[i][1] for i in range(15)]
    weight = torch.true_divide(torch.tensor(values), sum(values))
    crit = nn.CrossEntropyLoss(weight=weight.cuda())

    batch_size = 64
    traindl, valdl = data.BucketIterator.splits(datasets=(trainds, valds), 
                                            batch_size=batch_size, 
                                            sort_key=lambda x: len(x.text), 
                                            device=device, 
                                            sort_within_batch=True, 
                                            repeat=False)

    # print("[Corpus]: train: {}, test: {}, vocab: {}, labels: {}".format(
    #             len(train_iter.dataset), len(val_iter.dataset), len(TEXT.vocab), len(LABEL.vocab)))
    train_iter, val_iter = traindl, valdl
    TEXT, LABEL = text_field, label_field
    ntokens, nlabels = len(TEXT.vocab), len(LABEL.vocab)

    import math
    emsize = 300
    hidden = 64
    nlayers = 2
    lr = 1e-3
    clip = 0.25
    epochs = 10
    drop = 0.6
    model = 'LSTM'
    bi = True
    criterion = crit

    traindl, valdl = data.BucketIterator.splits(datasets=(trainds, valds), 
                                                batch_size=batch_size, 
                                                sort_key=lambda x: len(x.text), 
                                                device=device, 
                                                sort_within_batch=True, 
                                                repeat=False)

    train_iter, val_iter = traindl, valdl
    TEXT, LABEL = text_field, label_field

    embedding = nn.Embedding(ntokens, emsize, padding_idx=1, max_norm=1)
    if vectors: 
        embedding.weight.data.copy_(TEXT.vocab.vectors)
    encoder = Encoder(emsize, hidden, nlayers=nlayers, 
                    dropout=drop, bidirectional=bi)

    attention_dim = hidden if not bi else 2*hidden
    attention = Attention(attention_dim, attention_dim, attention_dim)

    model = Classifier(embedding, encoder, attention, attention_dim, nlabels, baseline=True).to(device)

    criterion = crit
    optimizer = torch.optim.Adam(model.parameters(), lr, amsgrad=True)

    for epoch in range(1, epochs + 1):
        m = train(epoch, model, train_iter, val_iter, optimizer, criterion)