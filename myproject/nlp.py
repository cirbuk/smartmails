#sentence length
#complex words with simple synonym
#
import spacy

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
simple = 0
total = 0
nlp = spacy.load('en')
text = nlp("There is a library you use to access the GPS hardware.")
analyzer(text)
for token in text:
	if (token.is_stop):
		simple += 1
	total += 1

if (simple * 2.4 < total):
	print("language is too complex, simplify")
print("analysis complete")


