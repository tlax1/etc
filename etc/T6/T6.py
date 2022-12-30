from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB

import pandas as pd

mess = pd.read_csv("NBC.csv", names=['message','label'])
mess['labelnum'] = mess.label.map({'pos': 1, 'neg': 0})

X_message = mess.message
Y_labelnum = mess.labelnum

X_train, X_test, Y_train, Y_test = train_test_split(X_message, Y_labelnum)

count_view = CountVectorizer()
X_train_f_tf = count_view.fit_transform(X_train)
X_test_tf = count_view.transform(X_test)

MNB = MultinomialNB()
MNB.fit(X_train_f_tf, Y_train)
Y_pred = MNB.predict(X_test_tf)

print('Accuracy: ', accuracy_score(Y_test, Y_pred))
print('Recall: ', recall_score(Y_test, Y_pred))
print('Precision: ', precision_score(Y_test, Y_pred))
print('Confusion Matrix: \n', confusion_matrix(Y_test, Y_pred))