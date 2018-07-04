from flask import Flask, render_template, request, jsonify, render_template
import nltk
import pickle
import json
import string
#import sklearn



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier

from nltk.sentiment.vader import SentimentIntensityAnalyzer
#uses the sentiment lexicon and morphological analysis to analyze sentences
#Must perform nltk.download('vader_lexicon')
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

#nlp1=spacy.load("tone_model")
obj_clf=pickle.load(open('training_models/subjectivity/subj_clf.joblib.pkl',"rb"), encoding = "latin1")
tone_clf=pickle.load(open('training_models/tone/tone_clf.joblib.pkl',"rb"), encoding = "latin1")
polite_clf=pickle.load(open('training_models/politeness/classifier.joblib.pkl',"rb"), encoding = "latin1")
print('Loaded SV classifier')

tf=pickle.load(open('training_models/subjectivity/vectorizer.joblib.pkl',"rb"), encoding = "latin1")
tf_tone=pickle.load(open('training_models/tone/vectorizer.joblib.pkl',"rb"), encoding = "latin1")
tf_polite=pickle.load(open('training_models/politeness/vectorizer.joblib.pkl', "rb"), encoding = "latin1")

print('vectorizer loaded')
wordslist = []
classlist = []

def getwordslist():
	file = open("subjwords.txt", "r")
	listwords = file.readlines()
	words = []
	classification = []
	for item in listwords:
		#print(item[5])
		if item[5] == "s":
			classification.append(item[5:15])
		if item[5] == "w":
			classification.append(item[5:13])
		items = item.split()
		words.append(items[2][6:])
	return words, classification

def buildhashtable(wordlist, classlist):
	dic = {}
	i = 0
	for word in wordlist:
		dic[word] = classlist[i]
		i += 1
	return dic



wordslist, classlist = getwordslist()
dic = buildhashtable(wordslist, classlist)
#print(dic)

@app.route('/<string:page_name>/')
def render_static(page_name):
	return render_template('%s.html' % page_name)



@app.route('/postmethod', methods = ['POST'])
def get_post_email_data():
	jsdata = request.form['data']
	#content=json.loads(jsdata)[0]
	#useless_words = nltk.corpus.stopwords.words("english") + list(string.punctuation)
	noisefreedata = removenoise(jsdata)
	word_count_length = word_count(noisefreedata)
	tokens = nltk.tokenize.TreebankWordTokenizer()
	tokenlist = tokens.tokenize(noisefreedata)
	resList = lemmatizeText(tokenlist)
	if resList==[]:
		return json.dumps({})
	print(resList)
	print('Called function')
	sentences= createSentenceList(noisefreedata)
	print(sentences)


	question_count_length = question_count(resList)
	processedData=getPunctFreeString(resList)

	#print(processedData)



	sid=SentimentIntensityAnalyzer()
	scores=sid.polarity_scores(processedData)
	obj_res,obj_score=obj_clf.predict(tf.transform([processedData])),obj_clf.predict_proba(tf.transform([processedData]))
	tone_res,tone_score=tone_clf.predict(tf_tone.transform([processedData])),tone_clf.predict_proba(tf_tone.transform([processedData]))
	polite_res, polite_score=polite_clf.predict(tf_polite.transform([processedData])), polite_clf.predict_proba(tf_polite.transform([processedData]))
	print(obj_score)
	obj_score_modified = modifysubjscore(processedData, obj_score, word_count_length)
	print(obj_score_modified)

	sentenceScores=getSentenceScores(sentences)
	getBadSentences(sentenceScores)
	print("Printing scores")
	i=0
	for item in sentenceScores:
		print(item,sentences[i])
		i+=1

	#print(subj_res[0])
	#print(subj_score)
	#print(tone_res[0])
	#print(polite_res[0])
	#print(polite_score)
	#print(tone_score)
	#doc1=nlp1(processedData.decode('utf-8'))
	complex_words_length, syllable_count = getComplexWords(resList)


	scores['word_count']=word_count_length
	scores['question_count']=question_count_length
	scores['complex_words']=complex_words_length
	scores['politeness']={'polite':round(polite_score[0, 0], 4), "rude": round(polite_score[0, 1], 4)}
	scores['subjectivity']=round(obj_score_modified[0,1],4)
	scores['objectivity']=round(obj_score_modified[0,0],4)
	scores['tone']={'anger':round(tone_score[0,0],4),'fear':round(tone_score[0,1],4),'joy':round(tone_score[0,2],4),'love':round(tone_score[0,3],4),'sadness':round(tone_score[0,4],4),'surprise':round(tone_score[0,5],4)}


	#print(scores)
	#sentiment = classifier.classify(build_bag_of_words(resList))
	#print(sentiment)
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
	finalstr = finalstr[:len(finalstr) - 1]
	return(finalstr)

def lemmatizeText(tokenlist):
	stemmer=nltk.stem.WordNetLemmatizer()
	for token in tokenlist:
		token = stemmer.lemmatize(token)
	return tokenlist

def word_count(str_text):
	text = str_text.split()
	length = 0
	for token in text:
		if token not in string.punctuation:
			length += 1
	return length   

def question_count(tokens):
	i = 0
	count = 0
	while i < len(tokens):
		if tokens[i] == "?":
			i += 1
			count += 1
			while i < len(tokens) and tokens[i] == "?":
				i += 1
		else:
			i += 1
	return count

def createSentenceList(text):
	sentences=[]
	currSentence=''
	punct=['.','!','?']
	i=0
	print("here it is")
	print(text[len(text)-1])
	if text[len(text) - 1] not in punct:
		text+= "."
		print("added punctuation")
		print(text)
	while i<len(text):

		if text[i] not in punct:
			currSentence+=text[i]
			i+=1
		else:
			sentences.append(currSentence)
			i+=1
			while i<len(text) and text[i] in punct:
				i+=1
			currSentence=''

	return sentences

def getSentenceScores(sentences):
	sid=SentimentIntensityAnalyzer()
	sentenceScores=[]
	for sentence in sentences:
		sentence_length=word_count(sentence)
		complex_words,syl=getComplexWords(sentence.split())
		scores=sid.polarity_scores(sentence)
		obj_res,obj_score=obj_clf.predict(tf.transform([sentence])),obj_clf.predict_proba(tf.transform([sentence]))
		tone_res,tone_score=tone_clf.predict(tf_tone.transform([sentence])),tone_clf.predict_proba(tf_tone.transform([sentence]))
		polite_res, polite_score=polite_clf.predict(tf_polite.transform([sentence])), polite_clf.predict_proba(tf_polite.transform([sentence]))
		obj_score_modified = modifysubjscore(sentence, obj_score,sentence_length)
		scores['word_count']=sentence_length
		scores['complex_words']=complex_words
		scores['politeness']={'polite':round(polite_score[0, 0], 4), "rude": round(polite_score[0, 1], 4)}
		scores['subjectivity']=round(obj_score_modified[0,1],4)
		scores['objectivity']=round(obj_score_modified[0,0],4)
		scores['tone']={'anger':round(tone_score[0,0],4),'fear':round(tone_score[0,1],4),'joy':round(tone_score[0,2],4),'love':round(tone_score[0,3],4),'sadness':round(tone_score[0,4],4),'surprise':round(tone_score[0,5],4)}
		sentenceScores.append(scores)
	return sentenceScores

def getBadSentences(sentenceScores):
	result = []
	i = 0
	while i < len(sentenceScores):
		errors = []
		if sentenceScores[i]['word_count'] > 15:
			errors.append("length")
		if sentenceScores[i]["complex_words"] > 3:
			errors.append("complexity")
		if sentenceScores[i]["neu"] > 0.65:
			errors.append("neutral")
		if sentenceScores[i]["neg"] > 0.6:
			errors.append("negative")	
		if sentenceScores[i]["politeness"]["polite"] < 0.5:
			errors.append("rude")
		if sentenceScores[i]["objectivity"] > 0.6:
			errors.append("objective")
		maximum = -1
		key = ''
		for k,v in sentenceScores[i]["tone"].items():
			if v > maximum:
				maximum = v
				key = k
		if key in ["anger", "fear", "sadness"]:
			errors.append(key)

		result.append((i, errors))
		i += 1
	print(result)

def getPunctFreeString(list):
	str1=''
	for word in list:
		if word not in string.punctuation:
			str1+=word+' '
	str1=str1[:len(str1)-1]
	return str1

def getComplexWords(text):
	ncomplex=0
	ntotal = 0
	for word in text:
		#word.__str__()
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
		ntotal += syllables
	return ncomplex, ntotal




def modifysubjscore(text, score, wordcount):
	subjlen = 0
	objlen = 0
	#print(dic)
	#print(text)
	words = []
	words = text.split()
	for word in words:
		if dic.get(word) == "strongsubj":
			subjlen += 1
		if dic.get(word) == "weaksubj":
			objlen += 1
	subj_ratio = float(subjlen) / wordcount
	obj_ratio = float(objlen) / wordcount
	subj_diff = abs(subj_ratio - obj_ratio)
	if subj_ratio > obj_ratio:
		score[0, 1] += subj_diff
		score[0, 0] -= subj_diff
	else:
		score[0, 0] += subj_diff
		score[0, 1] -= subj_diff

	if score[0,0] > 1:
		score[0, 0] = 1
		score[0, 1] = 0
	if score[0, 1] > 1:
		score[0, 1] = 1
		score[0, 0] = 0
	return score




if __name__ == '__main__':
	app.run(debug = True)