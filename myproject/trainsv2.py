import pickle
import spacy
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
nlp = spacy.load("en")
df1 = pd.read_csv("Stanford_politeness_corpus/wikipedia.annotated.csv", usecols = ["Request", "Classification"])
vectors = []
dataLabels = []
i = 1
for index, row in df1.iterrows():
	doc = nlp(row["Request"])
	vectors.append(doc.vector)
	dataLabels.append(row["Classification"])
	print(i)
	i += 1

xTrain = vectors[:3500]
yTrain = dataLabels[:3500]
xTest = vectors[3500:]
yTest = dataLabels[3500:]
print("training")

clf_svm=SGDClassifier(loss='modified_huber', penalty='l2',alpha=1e-3,max_iter=50,tol=None,random_state=42)
clf_svm = clf_svm.fit(X=xTrain, y=yTrain)
predict=clf_svm.predict(xTest)
print(len(predict))
print(accuracy_score(yTest,predict))
print("Trained SV classifier")
pickle.dump(clf_svm,open('politenessSVM/subj_clf.joblib.pkl',"wb"))

