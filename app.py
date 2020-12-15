import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
import pickle

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
model = pickle.load(open('encoder_test_lead_scoring.pkl', 'rb'))

with open("encoder_test_lead_scoring.pkl" , 'rb') as file:  
    enc = pickle.load(file)


@app.route('/')
def home():
    return render_template('index.html')
@cross_origin()
@app.route('/predict',methods=['POST'])
def predict():
    
    resp = request.get_json(force=True)
    print(resp)
    pred_df = (pd.DataFrame(resp)).reset_index()
    pred = enc.transform(pred_df).toarray()

    with open("../final_test_lead_scoring.pkl", 'rb') as file:  
        Pickled_ada_Model = pickle.load(file)


    #return str(Pickled_ada_Model.predict(pred)[0])
    print(pred)



if __name__ == "__main__":
    app.run(debug=True)
