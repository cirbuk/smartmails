from flask import Flask, render_template, request, jsonify
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#uses the sentiment lexicon and morphological analysis to analyze sentences
#Must perform nltk.download('vader_lexicon')
import json
import string
from flask_cors import CORS
app = Flask(__name__)
CORS(app)


@app.route('/postmethod', methods = ['POST'])
def get_post_email_data():
    jsdata = request.form['data']
    #content=json.loads(jsdata)[0]
    #useless_words = nltk.corpus.stopwords.words("english") + list(string.punctuation)
    noisefreedata = removenoise(jsdata)
    tokens = nltk.tokenize.TreebankWordTokenizer()
    tokenlist = tokens.tokenize(noisefreedata)
    resList = lemmatizeText(tokenlist)
    processedData=getPunctFreeString(resList)
    word_count_length = word_count(resList).__str__()
    sid=SentimentIntensityAnalyzer()
    scores=sid.polarity_scores(processedData)
    print(processedData)
    print(scores)
    #sentiment = classifier.classify(build_bag_of_words(resList))
    #print(sentiment)
    #eventually will return an object representing the results of analysis of different features/classes
    return jsdata + " word count is " + word_count_length + " sentiment and complexity scores" + json.dumps(scores)

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

if __name__ == '__main__':
    app.run(debug = True)