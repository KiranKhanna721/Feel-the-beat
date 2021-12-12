import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv('dataset.csv')
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])
df = df.drop('filename', axis = 1)

X = df.drop('label', axis = 1)
y = df.label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
clf = SVC(kernel='rbf')
clf.fit(X_train,y_train)
pred=clf.predict(X_test)
print(accuracy_score(pred,y_test))
pickle.dump(clf, open('model.pkl','wb'))