from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
import pickle
from random import shuffle
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
tf=TfidfVectorizer(analyzer='word')

def vectorizer(data):
    tfidf_matrix=tf.fit_transform(data)
    pickle.dump(tf,open('training_models/gender/vectorizer.joblib.pkl',"wb"),protocol=2)
    print('vectorizer saved')
    matrix=tfidf_matrix.toarray()
    return matrix


def trainSVClassifier():
    data = []
    data_labels = []
    data1=[]
    '''
    with open("./subobj_data/subjective.txt") as f:
        for i in f: 
            data.append(i) 
            data_labels.append('subj')
 
    with open("./subobj_data/objective.txt") as f:
        for i in f: 
            data.append(i)
            data_labels.append('obj')
'''
    niters=15000

    with open("./gender_data/maledata/male") as f:
        j=0
        for i in f:
            j+=1 
            sent[] = i.split()
            if len(sent) > 5:
                data.append((i,'male'))
            #data_labels.append('anger')
                if j==niters:
                    break
    with open("./gender_data/femaledata/female") as f:
        j=0
        for i in f:
            j+=1 
            sent[] = i.split()
            if len(sent) > 5:
                data.append((i,'female'))
            #data_labels.append('anger')
                if j==niters:
                    break
    
    print(len(data))
    shuffle(data)   
    for entry in data:
        print(entry)

        data1.append(entry[0])
        data_labels.append(entry[1])

    split=int(len(data) * 0.75)

    matrix=vectorizer(data1)
    X_train=matrix[:split]
    y_train=data_labels[:split]
    X_test=matrix[split:]
    y_test=data_labels[split:]
    print('Started training ')
    clf_svm=SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3,max_iter=50,tol=None,random_state=42)
    clf_svm = clf_svm.fit(X=X_train, y=y_train)
    predict=clf_svm.predict(X_test)
    print(len(predict))
    print(accuracy_score(y_test,predict))
    print("Trained SV classifier")
    pickle.dump(clf_svm,open('training_models/gender/gender_clf.joblib.pkl',"wb"),protocol=2)
trainSVClassifier()

'''anger-1 fear-2 joy-3 surprise-6 5-sadness 4-love
'''
