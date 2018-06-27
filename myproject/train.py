from pandas import DataFrame, read_csv
import pandas as pd
import spacy
from spacy.util import minibatch, compounding
from random import shuffle
nlp = spacy.load("en")

#df1 = pd.read_csv("newtrainingdata.csv", usecols = ["Request", "Classification"])

df1 = pd.read_csv("2classnewdata.csv", usecols = ["Request", "Classification"])
#print(df1.describe())

#df2 = pd.read_csv("stackpoliteness.csv", usecols = ["Request", "Classification"])
#print(df2.Request)

docs1 = []
docs2 = []
i = 0
#print(df1)


correct_categories = []

for index, row in df1.iterrows():
	docs1.append(row["Request"])
	#print(row["Classification"])
	#docs1[i].cats = row["Classification"]
	#print(docs1[i].cats)
	correct_categories.append(row["Classification"])
	print(i)
	i = i + 1


#print(correct_categories)

#print(docs1)


'''
for index, row in df2.iterrows():
	docs1.append(row["Request"])
	#print(docs2[i].cats)
	correct_categories.append(row["Classification"])
	print(i)
	i = i + 1
	'''

docs1_train = docs1
correct_categories_train = correct_categories

textcat = nlp.create_pipe("textcat")
nlp.add_pipe(textcat, last = True)

traincats = ["Polite", "Neutral", "Rude"]

for label in traincats:
	textcat.add_label(label)

dict_list = []

i = 0
for category in correct_categories_train:
	print(i)
	i = i + 1
	dict_cats = {"cats" : {}}
	temp = dict_cats.get("cats")
	for label in traincats:
		if label == category:
			temp[label] = 1
		else:
			temp[label] = 0
	dict_list.append(dict_cats)




train_data = list(zip(docs1_train, dict_list))
shuffle(train_data)
#print(train_data)
'''
optimizer = nlp.begin_training()
for itn in range(10):
	print(itn)
	for doc, gold in train_data:
		nlp.update([doc], [gold], sgd=optimizer)
doc = nlp(u'Hello thank you for the help')
print(doc.cats)
doc = nlp(u'Your work is terrible')
print(doc.cats)
'''
def evaluate(tokenizer, textcat, texts, cats):
	docs = (tokenizer(text) for text in texts)
	tp = 1e-8  # True positives
	fp = 1e-8  # False positives
	fn = 1e-8  # False negatives
	tn = 1e-8  # True negatives
	for i, doc in enumerate(textcat.pipe(docs)):
		gold = cats[i]
		for label, score in doc.cats.items():
			if label not in gold:
				continue
			if score >= 0.5 and gold[label] >= 0.5:
				tp += 1.
			elif score >= 0.5 and gold[label] < 0.5:
				fp += 1.
			elif score < 0.5 and gold[label] < 0.5:
				tn += 1
			elif score < 0.5 and gold[label] >= 0.5:
				fn += 1
	precision = tp / (tp + fp)
	recall = tp / (tp + fn)
	f_score = 2 * (precision * recall) / (precision + recall)
	return {'textcat_p': precision, 'textcat_r': recall, 'textcat_f': f_score}

# get names of other pipes to disable them during training




other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'textcat']
with nlp.disable_pipes(*other_pipes):  # only train textcat
	optimizer = nlp.begin_training()
	print("Training the model...")
	print('{:^5}\t{:^5}\t{:^5}\t{:^5}'.format('LOSS', 'P', 'R', 'F'))
	for i in range(5):
		print(i)
		losses = {}
			# batch up the examples using spaCy's minibatch
		batches = minibatch(train_data, size=compounding(4., 32., 1.001))


		for batch in batches:
			texts, annotations = zip(*batch)
			nlp.update(texts, annotations, sgd=optimizer, drop=0.2,
						   losses=losses)

		'''
		with textcat.model.use_params(optimizer.averages):
				# evaluate on the dev data split off in load_data()
			scores = evaluate(nlp.tokenizer, textcat, docs1_test, correct_categories_test)
		print('{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}'  # print a simple table
				  .format(losses['textcat'], scores['textcat_p'],
						  scores['textcat_r'], scores['textcat_f']))

'''	
nlp.to_disk("./2classmodel")
test_text = "Thank you for the help"
test_text2 = "your work was terrible"

doc = nlp(test_text)
doc2 = nlp(test_text2)

print(test_text, doc.cats)
print(test_text2, doc2.cats)


