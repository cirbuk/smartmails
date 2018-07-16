from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
tf=TfidfVectorizer(analyzer='word')

def vectorizer(data):
    tfidf_matrix=tf.fit_transform(data)
    pickle.dump(tf,open('training_models/politeness/vectorizer.joblib.pkl',"wb"), protocol=2)
    print('vectorizer saved')
    matrix=tfidf_matrix.toarray()
    return matrix


def trainSVClassifier():
    data = []
    data_labels = []
    df1 = pd.read_csv("politenessdata.csv", usecols = ["Request", "Classification"])
    docs1 = []
    i = 0

    for index, row in df1.iterrows():
        data.append(row["Request"])
        #print(row["Classification"])
        #docs1[i].cats = row["Classification"]
        #print(docs1[i].cats)
        data_labels.append(row["Classification"])
        print(i)
        i = i + 1

    matrix=vectorizer(data)
    X_train=matrix[:1624]
    y_train=data_labels[:1624]
    X_test=matrix[1624:]
    y_test=data_labels[1624:]
    clf_svm=SGDClassifier(loss='modified_huber', penalty='l2',alpha=1e-3,max_iter=50,tol=None,random_state=42)
    clf_svm = clf_svm.fit(X=X_train, y=y_train)
    predict=clf_svm.predict(X_test)
    print(len(predict))
    print(accuracy_score(y_test,predict))
    print("Trained SV classifier")
    pickle.dump(clf_svm,open('training_models/politeness/classifier.joblib.pkl',"wb"), protocol=2)

trainSVClassifier()