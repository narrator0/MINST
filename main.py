import pandas as pd
from sklearn import svm
import numpy as np

from data_process import process_train_data

train_data = pd.read_csv("data/train.csv")
features_train, labels_train, features_test, labels_test = process_train_data(train_data)

first_row = features_train.loc[0].values.reshape((28,28))

# print np.array2string(first_row, max_line_width=np.inf)
# print features_test

print "start training"
clf = svm.SVC()
clf.fit(features_train, labels_train)

print clf.predict(features_test[0:1])
print labels_test[0:1]
print clf.score(features_test, labels_test)
