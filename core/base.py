# packages & dependencies

from datasets import load_dataset			# load hugging face datasets
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import re

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV

import tokenizers                           # tokenizers from hugging face

# ************************************************************************

# load dataset from hugging face
# [huggingface data page](https://huggingface.co/datasets/presencesw/mednli) 

ds = load_dataset("presencesw/mednli")

# define training, test, validation split
train = load_dataset('presencesw/mednli', split = 'train')
test = load_dataset('presencesw/mednli', split = 'test')                    # save test sets as .csv
validation = load_dataset('presencesw/mednli', split = 'validation')


# convert to pd dataframe
train = train.to_pandas()
test = test.to_pandas()
validation = validation.to_pandas()


# save test set to csv
# test.to_csv('test.csv', index=False)

# data preview
train.head()

train.shape             # 11232 x 3
test.shape              # 1422 x 3
validation.shape        # 1395 x 3

train.columns

train['gold_label'].unique()     # true label
train['sentence1']               # premise
train['sentence2']               # hypothesis

# check training data is balanaced
Counter(train['gold_label'] == 'entailment')            # 3744 / 11232
Counter(train['gold_label'] == 'contradiction')         # 3744 / 11232
Counter(train['gold_label'] == 'neutral')               # 3744 / 11232


# pre-processing for traditional NLP models

encoding_map = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
decoding_map = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}

def preprocess(data):
    
    data['text'] = data['sentence1'] + ' ' + data['sentence2']    # combine
    data['text'] = data['text'].str.lower()                 # to lowercase
    
    data['text'] = data['text'].str.replace(r"[^\w\s]", "", regex=True)
    
    # encode target
    encoding_map = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
    data['label'] = data['gold_label'].map(encoding_map)
    
    return data[['label', 'text']]

train = preprocess(train)
validation = preprocess(validation)
test = preprocess(test)


# baseline random choice model
SEED = 100
np.random.seed(SEED)

random_predictions = np.random.choice(test['label'].unique(), size = test.shape[0])
classification_report(test['label'], random_predictions)


# TF-IDF + SVM
Xtrain = train['text']
ytrain = train['label']

Xval = validation['text']
yval = validation['label']

# term freq. vectorizer
vectorizer = TfidfVectorizer(ngram_range = (1, 3))
Xtrain_vec = vectorizer.fit_transform(train['text'])
Xval_vec = vectorizer.transform(validation['text'])

clf = LinearSVC(random_state = SEED)
clf.fit(Xtrain_vec, ytrain)

ypred = clf.predict(Xval_vec)

classification_report(yval, ypred)



# enhancing tfidf vectorizer
vectorizer = TfidfVectorizer(ngram_range = (1, 3),
                             max_df = 0.5
                             )
Xtrain_vec = vectorizer.fit_transform(train['text'])
Xval_vec = vectorizer.transform(validation['text'])

clf = SVC(random_state = SEED)
clf.fit(Xtrain_vec, ytrain)

ypred = clf.predict(Xval_vec)

classification_report(yval, ypred)


# gridsearch

from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('tf', TfidfVectorizer()),
    ('clf', SVC())
    ])

grid = {
    'tf__ngram_range': [(1,1), (1,2), (1,3)],
    'tf__min_df': [1, 2, 5],
    'tf__max_df': [0.75, 0.9, 1.0],
    'tf__sublinear_tf': [True],
    'tf__stop_words': ['english'],
    'clf__C': [0.01, 0.1, 1, 10],
    'clf__kernel': ['linear', 'poly', 'rbf']
    }

grid = RandomizedSearchCV(
    pipe,
    grid,
    cv=5,
    scoring='accuracy',
    verbose=1
    )

# fit on training data
grid.fit(train['text'], train['label'])

print(grid.best_params_)
'''
{'tf__sublinear_tf': True, 
'tf__stop_words': 'english', 
'tf__ngram_range': (1, 3), 
'tf__min_df': 1, 
'tf__max_df': 1.0, 
'clf__kernel': 'linear', 
'clf__C': 10}
'''

# term freq. vectorizer
vectorizer = TfidfVectorizer(ngram_range = (1, 3), 
                             sublinear_tf = True, 
                             stop_words = 'english',
                             min_df = 1,
                             max_df = 1.0,
                             )
Xtrain_vec = vectorizer.fit_transform(train['text'])
Xval_vec = vectorizer.transform(validation['text'])

clf = SVC(kernel = 'linear', C = 10, random_state = SEED)
clf.fit(Xtrain_vec, ytrain)

ypred = clf.predict(Xval_vec)

classification_report(yval, ypred)



# BoW + Logistic Regression
vectorizer = CountVectorizer(ngram_range = (1, 3))
Xtrain_vec = vectorizer.fit_transform(train['text'])
Xval_vec = vectorizer.transform(validation['text'])

clf = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
clf.fit(Xtrain_vec, ytrain)

ypred = clf.predict(Xval_vec)
classification_report(yval, ypred)



cm = confusion_matrix(yval, ypred)
classes = ['entailment', 'neutral', 'contradiction']
sb.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues',
           xticklabels = classes, yticklabels = classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for LR')
plt.show()


