from flask import Flask, jsonify ,render_template,url_for
# import pickle
import joblib as joblib
import sklearn
from flask import request as req
from textTransformer import transform_text

app = Flask(__name__)


# tfidf = pickle.load('vectorizer.pkl','rb')
# model = pickle.load('model.pkl','rb')
# with open('vectorizer.pkl', 'rb') as vectorizer_file:
#     tfidf = pickle.load(vectorizer_file)
#
# with open('model.pkl', 'rb') as model_file:
#     model = pickle.load(model_file)

tfidf = joblib.load('vectorizer2.pkl')
model = joblib.load('model2.pkl')


@app.route('/')
def index():
    return render_template('index.html')


@app.route("/detector",methods=['GET','POST'])
def detector():
    if req.method == "POST":
       data =req.form['data']

       transformed_msg = transform_text(data)
       vector_input=tfidf.transform([transformed_msg])
       result =model.predict(vector_input)[0]
       print(result)
       if result ==1:
           output = "Spam"
       else:
           output =  "Not Spam !"
       return render_template('index.html',result =output)
    return "Invalid request"
    
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
    



# app.run(debug=True)