import urllib2
import uuid
import csv
import json

import pandas as pd
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

f = csv.writer(open("/home/codephillip/Documents/fyp ml/weather.csv", "wb+"))
# smv = soil moisture value. comes from the smv sensor on the hardware
f.writerow(["pk", "dt", "name", "temp", "humidity", "smv", "trigger"])

# production url
# url = "http://api.openweathermap.org" + "/data/2.5/forecast?id=" + str(
#     233114) + "&mode=json&units=metric&cnt=7&appid=1f846e7a0e00cf8c2f96dd5e768580fb"
# development url
url = "http://127.0.0.1:8080/weather2.json"
print(url)
# 'load'-for json document, 'loads'-for json string
x = json.load(urllib2.urlopen(url))
print(x)
city = x.get('city')


def get_random_byte():
    if int(str(uuid.uuid4().get_time())[0:3]) > 500:
        return 0
    else:
        return 1


count = 1
for list_data in x.get('list'):
    f.writerow([count, list_data["dt_txt"],
                city["name"], list_data["main"]["temp"], list_data["main"]["humidity"],
                str(uuid.uuid4().get_node())[0:3],
                get_random_byte()])
    count += 1

# csv_url = "http://127.0.0.1:8080/test.csv"
# todo point to a global source for the csv file
# csv_url = "/home/codephillip/Documents/fyp ml/weather.csv"
csv_url = "weather.csv"
data = pd.read_csv(csv_url, index_col=0)
feature_cols = ['temp', 'humidity', 'smv']

print("ACTUAL DATA")
print(data.head())

X = data[feature_cols]
print(X.head())
print(type(X))
print(X.shape)

y = data['trigger']
print(y.head())
print(type(y))
print(y.shape)

# MACHINE LEARNING
# split the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
# default split is 75% for training and 25% for testing
print("train test split shapes")
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


def classifier1():
    print("LogisticRegression")
    # accuracy evaluation
    classifier = LogisticRegression()
    # fit the model to the training data (learn the coefficients)
    classifier.fit(X_train, y_train)
    # make predictions on the testing set
    y_pred = classifier.predict(X_test)
    print("y_test")
    print(y_test)
    print("y_pred")
    print(y_pred)
    print(metrics.accuracy_score(y_test, y_pred))


def classifier2():
    print("LinearSVC")
    # accuracy evaluation
    classifier = LinearSVC()
    # fit the model to the training data (learn the coefficients)
    classifier.fit(X_train, y_train)
    # make predictions on the testing set
    y_pred = classifier.predict(X_test)
    print("y_test")
    print(y_test)
    print("y_pred")
    print(y_pred)
    print(metrics.accuracy_score(y_test, y_pred))


def classifier3():
    print("RandomForestClassifier")
    # accuracy evaluation
    classifier = RandomForestClassifier(n_estimators=3, max_depth=2)
    # fit the model to the training data (learn the coefficients)
    classifier.fit(X_train, y_train)
    # make predictions on the testing set
    y_pred = classifier.predict(X_test)
    print("y_test")
    print(y_test)
    print("y_pred")
    print(y_pred)
    print(metrics.accuracy_score(y_test, y_pred))


def classifier4():
    print("RandomForestClassifier")
    # accuracy evaluation
    classifier = KNeighborsClassifier(n_neighbors=1)
    # fit the model to the training data (learn the coefficients)
    classifier.fit(X_train, y_train)
    # make predictions on the testing set
    y_pred = classifier.predict(X_test)
    print("y_test")
    print(y_test)
    print("y_pred")
    print(y_pred)
    # compare the predicted values-y_pred, with the actual values-y_test
    print(metrics.accuracy_score(y_test, y_pred))


# run all classifier to determine the best
classifier1()
classifier2()
classifier3()
classifier4()
