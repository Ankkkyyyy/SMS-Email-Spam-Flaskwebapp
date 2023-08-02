from flask import Flask, jsonify ,render_template
import pickle 
import nltk
# import jsonify
# nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import string
from flask import request as req 
app = Flask(__name__)
ps = PorterStemmer() 


# tfidf = pickle.load('vectorizer.pkl','rb')
# model = pickle.load('model.pkl','rb')
with open('vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf = pickle.load(vectorizer_file)

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

def transform_text(text):
    text = text.lower() # lowercase
    text = nltk.word_tokenize(text)  #Tokenization 
    y = []
    for i in text:
        if i.isalnum():
            y.append(i) 

    text = y[:] 
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:

            y.append(i)
    text = y[:] 
    y.clear()
    
    for i in text: 
        y.append(ps.stem(i))

    return " ".join(y)

@app.route('/')
def index():
    return render_template('index.html')


@app.route("/detector",methods=['GET',"POST"])
def detector():
    if req.method == "POST":
       data =req.form['data']

       transformed_msg = transform_text(data)
       vector_input=tfidf.transform([transformed_msg])
       result =model.predict(vector_input)[0]
       if result ==1:
           output = "Spam"
       else:
           output =  "Not Spam !"
       return render_template('index.html',result = output)
    
@app.route('/api/spamdetector', methods=['GET','POST'])
def spam_detector_api():
    if req.method == "POST":
        # data = req.form['text']
        data = req.json
        data = data['text']
    elif req.method == "GET":
        data = req.args.get('text', '')
    
    transformed_msg = transform_text(data)
    vector_input = tfidf.transform([transformed_msg])
    result = model.predict(vector_input)[0]
    
    if result == 1:
        output = "Spam"
    else:
        output = "Not Spam"

    return jsonify({"result": output})
    



app.run(debug=True)