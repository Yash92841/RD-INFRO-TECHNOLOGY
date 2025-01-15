import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

datapath = 'C:/Users/YASH/OneDrive/Desktop/spam sms classifier/data/spam_data.csv'
modelpath = 'C:/Users/YASH/OneDrive/Desktop/spam sms classifier/models/spam_classifier.pkl'
vectorizerpath = 'C:/Users/YASH/OneDrive/Desktop/spam sms classifier/models/tfidf_vectorizer.pkl'

try:
    data = pd.read_csv(datapath, encoding='latin-1')
except Exception as error:
    print(f"error loading csv file: {error}")
    raise

if 'block' not in data.columns or 'msg' not in data.columns:
    if len(data.columns) >= 2:
        data = data.iloc[:, :2]
        data.columns = ['block','msg']
    else:
        raise ValueError("dataset not have enough columns ")
    
data['block'] = data['block'].map({'ham':0,'spam':1})
data = data.dropna(subset=['msg'])
data['msg'] = data['msg'].astype(str)

xtrain, xtest, ytrain, ytest = train_test_split (
    data['msg'],data['block'], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words='english',max_features=5000)
xtrain_tfidf = vectorizer.fit_transform(xtrain)
xtest_tfidf = vectorizer.transform(xtest)

nbmodel = MultinomialNB()
nbmodel.fit(xtrain_tfidf, ytrain)

joblib.dump(nbmodel, modelpath)
joblib.dump(vectorizer, vectorizerpath)

loadedmodel = joblib.load(modelpath)
loadedvectorizer = joblib.load(vectorizerpath)

while True:
    userinput = input("enter message to check if it's spam or real or quit (ctrl + c) :")
    if userinput.lower() == 'exit':
        print("exiting")
        break

    newmessage_tfidf = loadedvectorizer.transform([userinput])

    prediction = loadedmodel.predict(newmessage_tfidf)
    print("spam" if prediction[0] == 1 else "real")
    