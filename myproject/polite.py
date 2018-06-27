import spacy
import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
nlp = spacy.load("en")
df1 = pd.read_csv("Stanford_politeness_corpus/wikipedia.annotated.csv", usecols = ["Request", "Normalized Score"])
vectors = []

i = 1
for index, row in df1.iterrows():
	doc = nlp(row["Request"])
	vectors.append(doc.vector)
	print(i)
	i += 1

print(len(vectors))
print(len(vectors[0]))
#print(vectors[0])

mu = np.mean(vectors, axis = 0)
sigma = np.cov(vectors.T)

p = multivariate_normal(mean=mu, cov=sigma)

test = "Have a nice day"
test2 = "put effort into your work"

