from flask import Flask, render_template, request, jsonify
import nltk
import pickle,spacy
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

nlp=spacy.load('newmodel')
#nlp1=spacy.load("tone_model")
obj_clf=pickle.load(open('model/subj_clf.joblib.pkl',"rb"), encoding = "latin1")
tone_clf=pickle.load(open('tone_model_1/subj_clf.joblib.pkl',"rb"), encoding = "latin1")
polite_clf=pickle.load(open('politenessmodel/classifier.joblib.pkl',"rb"), encoding = "latin1")
print('Loaded SV classifier')
tf=pickle.load(open('model/vectorizer.joblib.pkl',"rb"), encoding = "latin1")
tf_tone=pickle.load(open('tone_model_1/vectorizer.joblib.pkl',"rb"), encoding = "latin1")
tf_polite=pickle.load(open('politenessmodel/vectorizer.joblib.pkl', "rb"), encoding = "latin1")
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
	if resList==[]:
		return json.dumps({})
	print(resList)
	processedData=getPunctFreeString(resList)
	word_count_length = word_count(resList).__str__()
	sid=SentimentIntensityAnalyzer()
	scores=sid.polarity_scores(processedData)
	obj_res,obj_score=obj_clf.predict(tf.transform([processedData])),obj_clf.predict_proba(tf.transform([processedData]))
	tone_res,tone_score=tone_clf.predict(tf_tone.transform([processedData])),tone_clf.predict_proba(tf_tone.transform([processedData]))
	polite_res, polite_score=polite_clf.predict(tf_polite.transform([processedData])), polite_clf.predict_proba(tf_polite.transform([processedData]))
	print(tone_res[0])
	print(polite_res[0])
	print(polite_score)
	#print(tone_score)
	doc=nlp(processedData)
	#doc1=nlp1(processedData.decode('utf-8'))
	scores['complex_words']=getComplexWords(resList)
	scores['word_count']=word_count_length
	scores['subjectivity']=round(obj_score[0,1],4)
	scores['objectivity']=round(obj_score[0,0],4)
	#scores['politeness']={'polite':round(polite_score[0, 0], 4), "rude": }
	scores['tone']={'anger':round(tone_score[0,0],4),'fear':round(tone_score[0,1],4),'joy':round(tone_score[0,2],4),'love':round(tone_score[0,3],4),'sadness':round(tone_score[0,4],4),'surprise':round(tone_score[0,5],4)}
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