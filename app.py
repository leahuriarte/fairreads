from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import torch
import requests
import pickle
from nltk.stem import WordNetLemmatizer
import re
import newsapi
from newsapi import NewsApiClient


app = Flask(__name__)

#text extractor api
endpoint = "https://extractorapi.com/api/v1/extractor"
params = {
  "apikey": "d08a0c4577e2fa2c2f803842417b3881261a4371",
  "url": "example.com"
}
#news article collector api
newsapi = NewsApiClient(api_key='bdcec1609c0442e5b93945e12f3862aa')
#return redirect(url_for('upload'))


stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
             'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
             'does', 'doing', 'down', 'during', 'each','few', 'for', 'from', 
             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
             'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're',
             's', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
             'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 
             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
             'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
             'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
             "youve", 'your', 'yours', 'yourself', 'yourselves']

left = ["Vice News", "Buzzfeed", "​The Huffington Post", "Mashable", "MSNBC", "New York Magazine"]
lean_left = ["ABC News", "Al Jazeera English", "Associated Press", "Axios", "USA Today", "Time", "Bloomberg", "The Washington Post", "The Verge", "CBC News", "CBS News", "CNN", "ESPN", "AOL", "MTV News", "NBC News", "Politico", "Business Insider", "Bleacher Report", "Engadget", "Entertainment Weekly", "AOL", "Independent", "Infobae", "NRK", "Polygon", "Vox"]
center = ["Ars Technica", "Wired", "BBC News","The Wall Street Journal","The Jerusalem Post", "The Irish Times", "The Hill", "Focus", "The Globe And Mail", "Fortune", "National Geographic", "Newsweek", "Reuters", "Hacker News", "News24", "Recode", "Yahoo Entertainment", "Gothamist"]
lean_right = ["The Washington Times", "Financial Post", "Australian Financial Review", "News.com.au", "RBC"]
right = ["Breitbart News", "The American Conservative", "Fox News", "National Review"]

bias = "Unknown"
sentiment = "neutral"
title = "title"
articlelist = []

@app.route('/',methods=['GET', 'POST'])
def index():
    vectoriser, LRmodel = load_models() 
    #once user enters link
    if request.method == "POST":
        link = request.form.get("link") 
        params["url"] = link
        print(link)
        r = requests.get(endpoint, params=params)
        r= r.json()
        global title
        title = r["title"]
        sentences = r["text"].split(".")
        sentences = cleanup(sentences)
        predictions = predict(vectoriser, LRmodel, sentences)
        global sentiment
        sentiment = averager(predictions)
        all_articles = newsapi.get_everything(
            q=title,
            language='en', 
        )
        global articlelist
        articlelist = []
        for article in all_articles['articles']:
            global bias
            bias = "Unknown"
            if article['source']['name'] in left:
                bias = "Left"
            elif article['source']['name'] in lean_left:
                bias = "Leans Left"
            elif article['source']['name'] in center:
                bias = "Center"
            elif article['source']['name'] in lean_right:
                bias = "Leans Right"
            elif article['source']['name'] in right:
                bias = "Right"
            listy = [article['title'], article['source']['name'], bias, article['description'], article['url'], article['urlToImage']]
            print(listy)
            articlelist.append(listy)
        return render_template('upload.html', sentiment=sentiment, title1 = articlelist[0][0],  
                                title2 = articlelist[1][0], title3 = articlelist[2][0], source1 = articlelist[0][1],  
                                source2 = articlelist[1][1], source3 = articlelist[2][1], bias1 = articlelist[0][2],  
                                bias2 = articlelist[1][2], bias3 = articlelist[2][2], des1 = articlelist[0][3],  
                                des2 = articlelist[1][3], des3 = articlelist[2][3], url1 = articlelist[0][4], 
                                url2 = articlelist[1][4], url3 = articlelist[2][4], img1 = articlelist[0][5], 
                                img2 = articlelist[1][5], img3 = articlelist[2][5])
    return render_template('index.html')


def cleanup(sentences):
    for i in range(len(sentences)):
        sentences[i] = sentences[i].strip()
    sentences = [i for i in sentences if i != '']
    return sentences

def preprocess(sentences):
    processedText = []
    wordLemm = WordNetLemmatizer()
    urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    userPattern       = '@[^\s]+'
    alphaPattern      = "[^a-zA-Z0-9]"
    sequencePattern   = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"
    
    for sentence in sentences:
        sentence = sentence.lower()
        #replacing urls, @ symbols, non-alphabet characters
        sentence = re.sub(urlPattern,' URL',sentence)
        sentnece = re.sub(userPattern,' USER', sentence)        
        sentence = re.sub(alphaPattern, " ", sentence)
        sentence = re.sub(sequencePattern, seqReplacePattern, sentence)

        sentencewords = ''
        for word in sentence.split():
            #check for stopword
            #if word not in stopwordlist:
            if len(word)>1:
                # lemmatize
                word = wordLemm.lemmatize(word)
                sentencewords += (word+' ')
            
        processedText.append(sentencewords)
    return processedText

def load_models():
    file = open('vectoriser.pickle', 'rb')
    vectoriser = pickle.load(file)
    file.close()
    file = open('Sentiment-LR.pickle', 'rb')
    LRmodel = pickle.load(file)
    file.close()
    return vectoriser, LRmodel

def predict(vectoriser, model, text):
    textdata = vectoriser.transform(preprocess(text))
    sentiment = model.predict(textdata)
    
    data = []
    for text, pred in zip(text, sentiment):
        data.append((text,pred))  
    return data

def averager(data):
    count = 0
    points = 0
    for prediction in data:
        count += 1
        points += prediction[1]
    points = points/count
    if points < 0.3:
        return "Heavily Negative"
    elif points < 0.4:
        return"Slightly Negative"
    elif points < 0.6:
        return"Neutral"
    elif points < 0.7:
         return "Slightly Positive"
    else:
        return "Heavily Positive"