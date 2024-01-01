import json
from flask import Flask, request, redirect
# from sklearn.externals 
import joblib
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import ast
import pickle
from typing import Dict, List
import pprint
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re


app = Flask(__name__)

categories: List[str] = ["cs.AI", "cs.CL", "cs.CV", "cs.LG", "cs.NE", "cs.RO", "eess.IV", "Others", "stat.ML" ]

svc_models: Dict[str, object] = dict.fromkeys(categories, '')
for cat in categories:
    svc_models[cat] = joblib.load("svc_pipeline_model_{}.pkl".format(cat))

tfidf_vect = pickle.load(open("vectorizer.pickle", "rb"))
pp = pprint.PrettyPrinter(indent=4)


stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"])

re_stop_words = re.compile(r"\b(" + "|".join(stopwords) + ")\\W", re.I)

stemmer = SnowballStemmer("english")
def stemming(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence

def decontract(sentence):
    # specific
    sentence = re.sub(r"won't", "will not", sentence)
    sentence = re.sub(r"can\'t", "can not", sentence)

    # general
    sentence = re.sub(r"n\'t", " not", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"\'s", " is", sentence)
    sentence = re.sub(r"\'d", " would", sentence)
    sentence = re.sub(r"\'ll", " will", sentence)
    sentence = re.sub(r"\'t", " not", sentence)
    sentence = re.sub(r"\'ve", " have", sentence)
    sentence = re.sub(r"\'m", " am", sentence)
    return sentence

def cleanPunc(sentence): 
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    return cleaned

def keepAlpha(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', '', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent

def removeStopWords(sentence):
    global re_stop_words
    return re_stop_words.sub("", sentence)

def prepare_text(text: str) -> str:
    text = text.lower()
    text = decontract(text)
    text = cleanPunc(text)
    text = keepAlpha(text)
    text = removeStopWords(text)
    text = stemming(text)
    return text

def classify(class_request: Dict[str, str]) -> str: 
    # test_str = "{    'summaries': 'stereo match one wide use techniqu infer depth'}"
    # query_df = pd.DataFrame(test_str)
    # target.update({"summaries":"some data"})
    # df_dict: Dict(str, [str]) = { 'summaries': [ 'stereo match one wide use techniqu infer depth']}
    # FROM https://stackoverflow.com/questions/53465114/storing-tfidfvectorizer-for-future-use
    # content=jsonify(request.json)
    # test = pd.io.json.json_normalize(request.json)
    # tfidf_vect = pickle.load(open("vectorizer.pickle", "rb"))
    # test['ingredients'] = [str(map(makeString, x)) for x in test['ingredients']]
    # test_transform = tfidf_vect.transform(test['ingredients'].values)
    # le = preprocessing.LabelEncoder()
    # X_test = test_transform
    # y_test = le.fit_transform(test['cuisine'].values)

    # df_dict: Dict(str, [str]) = { 'summaries': pd.Series( class_request["summaries"])}
    prepared_text:str = prepare_text(class_request["summaries"])
    list_of_one: List(str) = [prepared_text]
    test_transform = tfidf_vect.transform(list_of_one)
    print("vectorised text:: getnnz=",test_transform.getnnz())
    print(test_transform)

    # query_df = pd.DataFrame(data=df_dict)

    # query = pd.get_dummies(query_df)
    predict_results: List(Dict[str,str]) = list()
    for cat in categories:
        result: Dict[str,str] = dict()
        result["category"] = cat
        result["is_classified"] = str(svc_models[cat].predict(test_transform)[0])
        result["decision_function"] = str(svc_models[cat].decision_function(test_transform)[0])
        predict_results.append(result)

    pp.pprint(predict_results)
    return predict_results


@app.route("/")
def hello():
    return "Welcome to machine learning model APIs!"

@app.route("/api/classification", methods=['POST'])
def api():
    class_request: Dict[str, str] = request.get_json()
    output_classifications = classify(class_request)
    return json.dumps(output_classifications, indent = 4) 


# @app.route('/classification')
# def classify_request():
#     text = "stereo match one wide use techniqu infer depth"
#     output_classifications = classify(text)
    # abstract_json = request.json

if __name__ == '__main__':
    app.run(debug=True, port=5000)



# POST /classifier
#    { abstract: ""}
# ==>>
# 200 [ "", ""]