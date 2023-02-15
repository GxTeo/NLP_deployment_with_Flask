from flask import Flask, request,render_template
from functions import PreProcess
import pickle
import re
from joblib import load
import numpy as np
import pandas as pd
import contractions
import emoji
import time
import string 
import os
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression


# Load the trained models from the pkl files
naive_bayes_model = load("../TF_IDF_models/Naive_Bayes_TFIDF_model.pkl")
linear_svc_model = load("../TF_IDF_models/Linear_SVC_TFIDF_model.pkl")
logistic_regression_model = load("../TF_IDF_models/Logistic_Regression_TFIDF_model.pkl")
# Create a Flask application

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    predictions = ""
    if request.method == 'POST':
        input_review = request.form['inputText']
        chosen_model = request.form['Model']
    if(input_review == ""):
        predictions = "Empty String"
        return render_template('index.html',input_text = input_review, predictions=predictions)
    if chosen_model == 'Naive Bayes':
        model = naive_bayes_model
    elif chosen_model == 'Support Vector Machine':
        model = linear_svc_model
    elif chosen_model == 'Logistic Regression':
        model = logistic_regression_model
    else:
        return "Error, invalid model"
    
    func = PreProcess()
    clean_review = func.clean_text(input_review)
    print("Clean Review: ", clean_review)
    train = pd.read_csv("../dataset/x_train.csv")
    X_train, x_test, Y_train, y_test = train_test_split(train["concat_review"], train["polarity"], test_size=0.2, random_state=42)
    # Tf-Idf representation
    tfidf_vect = TfidfVectorizer(min_df=5, max_features=10000, ngram_range=(1,2), lowercase=False, tokenizer=word_tokenize)
    X_train_tfidf = tfidf_vect.fit_transform(X_train)
    clean_review = tfidf_vect.transform([clean_review])
    predictions = model.predict(clean_review)[0]
    if(predictions==1):
        predictions="Positive"
    elif(predictions==0):
        predictions="Negative"
    return render_template('index.html',input_text = input_review, predictions=predictions)

# Start the Flask application
if __name__ == '__main__':
    app.run(debug=True)