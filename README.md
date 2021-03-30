# Reddit Post Flair Classifier
Classifies Reddit posts into their flairs
model based on LSTM self attention

run ```main.py``` 

**Performance:** f1 score of 0.55 , 0,26 (micro, macro) on validation set

**Depencies:** torch (preferably compiled with cuda), torchtext, numpy , pandas, spacy

**performance analysis:** Baseline (without attention) performs slighly better than with attention in the begining but gets better with epochs
