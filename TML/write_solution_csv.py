# import libraries
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
nltk.download('omw-1.4')
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
import re
import emoji


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

def change_label(label):
    if label == 0:
        return 'not sexist'
    else:
        return 'sexist'


train_df = pd.read_csv('nlp2/data/train_all_tasks.csv')
train_df['text'] = train_df['text'].apply(lambda x: preprocessing(x))
train_texts = train_df.text.values
train_labels = train_df.label_sexist.values


test_df = pd.read_csv('nlp2/data/dev_task_a_entries.csv')
test_df['text'] = test_df['text'].apply(lambda x: preprocessing(x))
test_ids = test_df.rewire_id.values
test_texts = test_df.text.values

vect = TfidfVectorizer(max_features=3000)
#vect = CountVectorizer(ngram_range=(1,3))
vect.fit(train_texts)
vect_train_texts = vect.transform(train_texts)
vect_test_texts = vect.transform(test_texts)

#model = LogisticRegression(penalty = 'l2')
model = SVC(C = 2, kernel = 'linear')
model.fit(vect_train_texts, train_labels)
predicted = model.predict(vect_test_texts)

file = open('nlp2/results/SUBMISSION_SVC_tfidf.csv', 'w')
file.write('rewire_id,label_pred\n')

for i in range(len(predicted)):
    text = test_ids[i] + ',' + predicted[i] + '\n'
    file.write(text)

file.close()
