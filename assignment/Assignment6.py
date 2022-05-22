# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Iris.csv')

X = dataset.iloc[:, :4].values
y = dataset['Species'].values
print(dataset)

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Training the Naive Bayes Classification model on the Training Set

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(y_pred)
df = pd.DataFrame({'Real Values': y_test, 'Predicted Values': y_pred})
print(df)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)
from sklearn.metrics import accuracy_score

print("Accuracy : ", accuracy_score(y_test, y_pred))
y_pred = classifier.predict(X_test)
print(y_pred)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import precision_score

# precision =precision_score(y_test, y_pred,average='macro')
print("precision : ", precision_score(y_test, y_pred, average='macro'))

y_pred = classifier.predict(X_test)
print(y_pred)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import recall_score

# precision =precision_score(y_test, y_pred,average='macro')
print("recall : ", recall_score(y_test, y_pred, average='macro'))
