import pandas as pd
import unicodedata
import re
import spacy
import torchtext
from torchtext import data
from torchtext import vocab
from torchtext.vocab import Vectors, GloVe
import os

class dataload():
    def __init__(self):
        self.test = pd.read_csv("NLP/NLP Data/data/val.csv")
        self.train = pd.read_csv("NLP/NLP Data/data/train.csv")
        self.train['title'] = self.train['Title'].str.cat(self.train['Post'],sep=" ")
        self.test['title'] = self.test['Title'].str.cat(self.test['Post'],sep=" ")
        self.nlp = spacy.load('en',disable=['parser', 'tagger', 'ner'])

    def tokenizer(self,s): 
        return [w.text.lower() for w in self.nlp(s)]

    def clean (self):
        cols = ['Post', 'Title']
        tr, te = self.train.drop(cols, axis=1), self.test.drop(cols, axis=1)
        tr.columns = ['label', 'text']
        te.columns = ['label', 'text']
        tr=tr.reindex(columns=['text', 'label'])
        te=te.reindex(columns=['text', 'label'])
        os.mkdir('./dataset')
        tr.to_csv('./dataset/train.csv', index=False);te.to_csv('./dataset/val.csv', index=False)

    def load_pretrained_vectors(self, dim):
        if dim in [50, 100, 200, 300]:
            name = 'glove.{}.{}d'.format('6B', str(dim))
            return name
        return None

    def buildVocab(self): 
        self.clean()
        label_field = data.Field(sequential=False, use_vocab=False, pad_token=None, unk_token=None)
        text_field = data.Field(sequential=True, tokenize=self.tokenizer, include_lengths=True, use_vocab=True)

        train_val_fields = [
            ('text', text_field),
            ('label', label_field)
        ]

        trainds, valds = data.TabularDataset.splits(path='./', format='csv', train='dataset/train.csv', 
                        validation='dataset/val.csv', fields=train_val_fields, skip_header=True)
        vectors = self.load_pretrained_vectors(300)

        text_field.build_vocab(trainds, valds, max_size=100000, vectors=vectors)
        label_field.build_vocab(trainds)

        return text_field, label_field, trainds, valds, vectors
