from flask import Flask, render_template, request, jsonify
import spacy
import nltk
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
@app.route('/', methods = ['GET', 'POST'])
def index():
	#print("hello")
	emaildata = request.get_json()
	#print(emaildata)
	#print(emaildata.get("email"))
	recievingemail = emaildata.get("email")
	idemail = recievingemail.get("thread_id")

	threademail = recievingemail.get("threads")
	content = threademail.get(idemail)
	contentplain = content.get("content_plain")
	print(contentplain)

	

	#content = threademail.get(idemail)
	#print(content)
	#contentplain = content.get("content_plain")

	#content = threademail[0]
	#print(content.get("content_plain"))
	#print(contentplain)
	#print(threademail.get("content_plain"))
	#print(request.args)
	#print(emaildata)

	simple = 0
	total = 0
	nlp = spacy.load('en')
	text = nlp("There is a library you use to access the GPS hardware. If you're working with a lot of text, you'll eventually want to know more about it.")	

	for token in text:
		if (token.is_stop):
			simple += 1
		total += 1
	
	length = 0
	punct = [".", ",", "?", "!"]
	numwrong = 0
	for token in text:
		if token.__str__() not in punct:
			length += 1
		if token.__str__() == ".": 
			#print(length)
			if (length > 12):
				numwrong += 1
	toolong = numwrong.__str__()
	if numwrong.__str__() != "0":
		error1 = " " + numwrong.__str__() + " Sentences are too long, shorten them."


	if (simple * 2 < total):
		complexity = True
		error2 = " Language is too complex, simplify."


	data = {
		"too long" : toolong, "complex" : complexity
	}
	#return request.args
	return jsonify(data)

def stringReturn(s):
	return s

def replaceadp(doc):
	words = []
	for token in doc:

		words.append(token)
		#print("check if works")
		#print(token.pos_)
	length = len(words)
	i = 0
	while i < length - 1:
		if words[i].text == "," and (words[0].pos_ == "CCONJ"):
			print("delete the word '" + words[0].text + "' and make 2 sentences")
			return True
		if words[i].text == "," and (words[i + 1].pos_ == "CCONJ"):
			print("split at the word '" + words[i + 1].text + "'")
			newsent = []
			newsent = words[:i] + ["."] + words[i + 2 :]
			print("new sentence:")
			return True
		i += 1
	return False

def analyzer(d):
	length = 0
	punct = [".", ",", "?", "!"]
	numwrong = 0
	for token in d:

		if token.__str__() not in punct:
			length += 1
		if token.__str__() == ".": 
			#print(length)
			if (length > 15):
				numwrong += 1
	if numwrong.__str__() != "0":
		print(numwrong.__str__() + " sentences are too long, shorten them")
				#if not replaceadp(d):
					#print("text is fine")



if __name__ == '__main__':
	app.run(debug = True)
