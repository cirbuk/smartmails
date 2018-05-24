import nltk
from sklearn.feature_extraction.text import TfidfVectorizer 
import pandas as pd


#tokenization, stemming, lemmatization
text = "feet cats wolves talked"
tokenizer = nltk.tokenize.TreebankWordTokenizer()
tokens = tokenizer.tokenize(text)


stemmer = nltk.stem.PorterStemmer()
newtokens = " ".join(stemmer.stem(token) for token in tokens)
print(newtokens)

stemmer = nltk.stem.WordNetLemmatizer()
newtokens2 = " ".join(stemmer.lemmatize(token) for token in tokens)
print(newtokens2)

#TF-IDF
texts = ["good movie", "not a good movie", "did not like"]
tfidf = TfidfVectorizer(min_df = 2, max_df = 0.5, ngram_range = (1, 2))
features = tfidf.fit_transform(texts)
pd.DataFrame(
	features.todense()
	#tfidf.get_feature_names()
	)
for elem in features:
	print("next")
	print(elem)
