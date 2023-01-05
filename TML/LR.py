# import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
import re
import emoji
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('omw-1.4')
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def preprocessing(text):

    lemma = WordNetLemmatizer()
    swords = stopwords.words("english")
    text = emoji.replace_emoji(text)
    text = re.sub("[^a-zA-Z0-9]"," ",text)
    text = re.sub(' +', ' ', text)

    text = nltk.word_tokenize(text.lower())
    text = [lemma.lemmatize(word) for word in text]

    text = [word for word in text if word not in swords]

    text = " ".join(text)
    return text 


task = 'label_sexist'


train_df = pd.read_csv('nlp2/data/train_all_tasks.csv')
train_df['text'] = train_df['text'].apply(lambda x: preprocessing(x))


if task == 'label_sexist':
    train_df[task] = train_df[task].apply(lambda x: 0 if x == 'not sexist' else 1)
else:
    train_df = train_df[train_df['label_sexist'] == 'sexist']
    train_df[task] = train_df[task].apply(lambda x: int(x[0]) - 1)


train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df['text'].values.tolist(), train_df[task].values.tolist(),
    stratify = train_df[task], test_size=0.2
)

penalty_param = ['l2', 'none']
max_features_tfidf_param = [3000, 5000, 8000]
n_gram_param = [1,2,3]

scores = []

for penalty in penalty_param:
    for max_features in max_features_tfidf_param:
        vect = TfidfVectorizer(max_features=max_features)
        vect.fit(train_texts)
        vect_train_texts = vect.transform(train_texts)
        vect_val_texts = vect.transform(val_texts)

        model = LogisticRegression(penalty = penalty)
        model.fit(vect_train_texts, train_labels)

        predicted = model.predict(vect_val_texts)
        score = accuracy_score(predicted, val_labels)
        text = 'Params: TFIDF max features={}, penalty={}, Acc: {}'.format(max_features, penalty, score)
        scores.append(text)
    

    for n_gram in n_gram_param:
        vect = CountVectorizer(ngram_range=(1, n_gram))
        vect.fit(train_texts)
        vect_train_texts = vect.transform(train_texts)
        vect_val_texts = vect.transform(val_texts)

        model = LogisticRegression(penalty = penalty)
        model.fit(vect_train_texts, train_labels)

        predicted = model.predict(vect_val_texts)
        score = accuracy_score(predicted, val_labels)
        text = 'Params: CountVectorizer n_gram={}, penalty={}, Acc: {}'.format(n_gram, penalty, score)
        scores.append(text)
        


scores = sorted(scores)
for score in scores:
    print(score)