from flask import Flask, render_template, request, jsonify
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#uses the sentiment lexicon and morphological analysis to analyze sentences
#Must perform nltk.download('vader_lexicon')
import json
import string
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

#tf=TfidfVectorizer(analyzer='word')

def vectorizer(data):
	tfidf_matrix=tf.fit_transform(data)
	joblib.dump(tf,'model/vectorizer.joblib.pkl',compress=9)
	print('vectorizer saved')
	matrix=tfidf_matrix.toarray()
	return matrix


def trainSVClassifier():
	data = []
	data_labels = []
	with open("./subobj_data/subj.txt") as f:
		for i in f: 
			data.append(i) 
			data_labels.append('subj')
 
	with open("./subobj_data/obj.txt") as f:
		for i in f: 
			data.append(i)
			data_labels.append('obj')
	matrix=vectorizer(data)

	'''matrix=vectorizer(data)
	X_train=matrix
	y_train=data_labels
	#X_test=matrix[8000:]
	#y_test=data_labels[8000:]
	clf_svm=SGDClassifier(loss='modified_huber', penalty='l2',alpha=1e-3,max_iter=50,tol=None,random_state=42)
	clf_svm = clf_svm.fit(X=X_train, y=y_train)
	#predict=clf_svm.predict(X_test)
	#print(len(predict))
	#print(accuracy_score(y_test,predict))
	print("Trained SV classifier")
	joblib.dump(clf_svm,'model/subj_clf.joblib.pkl',compress=9)'''

	


obj_clf=joblib.load('model/subj_clf.joblib.pkl')
print('Loaded SV classifier')
tf=joblib.load('model/vectorizer.joblib.pkl')
print('vectorizer loaded')


@app.route('/postmethod', methods = ['POST'])
def get_post_email_data():
	jsdata = request.form['data']
	#content=json.loads(jsdata)[0]
	#useless_words = nltk.corpus.stopwords.words("english") + list(string.punctuation)
	noisefreedata = removenoise(jsdata)
	tokens = nltk.tokenize.TreebankWordTokenizer()
	tokenlist = tokens.tokenize(noisefreedata)
	resList = lemmatizeText(tokenlist)
	print(resList)
	processedData=getPunctFreeString(resList)
	word_count_length = word_count(resList).__str__()
	sid=SentimentIntensityAnalyzer()
	scores=sid.polarity_scores(processedData)
	obj_res,obj_score=obj_clf.predict(tf.transform([processedData])),obj_clf.predict_proba(tf.transform([processedData]))
	print(obj_res[0])
	print(obj_score)
	scores['complex_words']=getComplexWords(resList)
	scores['word_count']=word_count_length
	scores['subjectivity']=round(obj_score[0,1],4)
	scores['objectivity']=round(obj_score[0,0],4)
	print(scores)
	#sentiment = classifier.classify(build_bag_of_words(resList))
	#print(sentiment)
	#eventually will return an object representing the results of analysis of different features/classes
	return json.dumps(scores)

def removenoise(input):
	l=input.split()
	res=[]
	for string in l:
		res+=string.split('<')
	res1=[]
	for string in res:
		res1+=string.split('>')
	res2=[]
	for string in res1:
		res2+=string.split('&')
	noise=['div','/div','br','nbsp;']
	result=[x for x in res2 if x not in noise and x]
	finalstr=''
	for string in result:
		finalstr+=string+' '
	return(finalstr)

def lemmatizeText(tokenlist):
	stemmer=nltk.stem.WordNetLemmatizer()
	for token in tokenlist:
		token = stemmer.lemmatize(token)
	return tokenlist

def word_count(text):
	length = 0
	for token in text:
		if token not in string.punctuation:
			length += 1
	return length   

def getPunctFreeString(list):
	str1=''
	for word in list:
		if word not in string.punctuation:
			str1+=word+' '
	str1=str1[:len(str1)-1]
	return str1

def getComplexWords(text):
	ncomplex=0
	for word in text:
		word.__str__()
		syllables = 0
		for i in range(len(word)):
			if i == 0 and word[i] in "aeiouy" :
				syllables = syllables + 1
			elif word[i - 1] not in "aeiouy" :
				if i < len(word) - 1 and word[i] in "aeiouy" :
					syllables = syllables + 1
				elif i == len(word) - 1 and word[i] in "aiouy" :
					syllables = syllables + 1
		if len(word) > 0 and syllables == 0 :
			syllables = 1
		if syllables>=3:
			ncomplex=ncomplex+1
	return ncomplex


if __name__ == '__main__':
	app.run(debug = True)