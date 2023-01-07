# SemEval2023
## Task 10: Towards Explainable Detection of Online Sexism

### Preprocessing:
- Convert hashtags (#HappyBirthday -> happy birthday)
- Replace emoji with description
- Tokenize -- data is taken from Gab + Reddit. Is TweetTokenizer better than NLTK?
- Lowercase all
- Remove punctuation
- Remove stop words
- Stem words -- WordNet stemmer (NLTK) requires POS-pairs. Better than Porter & Lancaster as they destroy too many sentiment distinctions
- To try: Negation marking (Das and Chen 2001, Pang et al. 2002) -- Append a _NEG suffix to every word appearing between a negation and a clause-level punctuation mark
    ex 1: no one*_NEG* enjoys*_NEG* it*_NEG*.
    ex 2:I don't think*_NEG* i*_NEG* will*_NEG* enjoy*_NEG* it*_NEG*, but I might. 


### Embeddings:
- TfIdf vs. Word2vec vs. GloVe


### RNN:
- GRU vs. LSTM vs. BiLSTM


### Transformers:
- DynaSent: http://www.github.com/cgpotts/dynasent


Link colab SVM: https://colab.research.google.com/drive/19OHkv6H9z5aaU5GX8LUoo3TR2k9AzlUb?usp=sharing
Link colab RNN: https://colab.research.google.com/drive/1b4OIYYuPSfq008xDDTOV2N1KYg9cAJ3o?usp=sharing
