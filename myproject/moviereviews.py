import nltk
import string
#nltk.download("movie_reviews")
#nltk.download("stopwords")
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier

#print(movie_reviews.fileids()[:5])
positive_fileids = movie_reviews.fileids("pos")
negative_fileids = movie_reviews.fileids("neg")
#print(movie_reviews.words(fileids = positive_fileids[0]))
useless_words = nltk.corpus.stopwords.words("english") + list(string.punctuation)

def build_bag_of_words(words):
	return {word : 1 for word in words if not word in useless_words}

negative_features = [(build_bag_of_words(movie_reviews.words(fileids = [f])), "neg") for f in negative_fileids]
positive_features = [(build_bag_of_words(movie_reviews.words(fileids = [f])), "pos") for f in positive_fileids]
#print(negative_features[0])
#print(positive_features[0])

split = 800
sentiment_classifier = NaiveBayesClassifier.train(positive_features[:split] + negative_features[:split])

print(nltk.classify.util.accuracy(sentiment_classifier, positive_features[:split] + negative_features[:split]))
print(nltk.classify.util.accuracy(sentiment_classifier, positive_features[split:] + negative_features[split:]))
print(sentiment_classifier.show_most_informative_features())
