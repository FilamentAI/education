import csv

from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

X=[]
Y=[]

#Load data into arrays
with open('spam.csv') as f:
    f.readline()#ignore first line
    reader=csv.reader(f,delimiter=',')
    for rows in reader:
        X.append(rows[1])
        Y.append(rows[0])

# preprocess, vectorize X values.
CV=CountVectorizer().fit_transform(X)


Xtrain, Xtest, Ytrain, Ytest = train_test_split(CV, Y, test_size=0.33)

#Build a naive Bayes model.
model = MultinomialNB()
model.fit(Xtrain, Ytrain)

#Build a neural network model.
model2 = MLPClassifier()
model2.fit(Xtrain, Ytrain)

#Build a decission tree.
model3 = DecisionTreeClassifier()
model3.fit(Xtrain, Ytrain)

preds = model.predict(Xtest)
preds2 = model2.predict(Xtest)
preds3 = model3.predict(Xtest)

#Generate classification report.
from sklearn.metrics import classification_report
print("This is the naive bayes classifier: ")
print(classification_report(Ytest,preds))
print("This is the neural network classifier: ")
print(classification_report(Ytest, preds2))
print("This is for the Decision Tree: ")
print(classification_report(Ytest, preds3))
