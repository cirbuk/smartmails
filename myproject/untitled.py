import pickle
tone_clf=pickle.load(open('training_models/tone/tone_clf.joblib.pkl',"rb"), encoding = "latin1")
tf_tone=pickle.load(open('training_models/tone/vectorizer.joblib.pkl',"rb"), encoding = "latin1")
pickle.dump(tf_tone,open('training_models/tone/vectorizer.joblib.pkl',"wb"), protocol=2)
pickle.dump(tone_clf,open('training_models/tone/tone_clf.joblib.pkl',"wb"), protocol=2)
