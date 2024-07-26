from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
from ai_model import reg

from flask import Flask,request, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
 int_features=[float(x) for x in request.form.values()]
 

 features=[np.array(int_features)]
 prediction= reg.predict(features)
 output= round(prediction[0],2)
 return render_template('index.html',prediction_text='ach sales {}'.format(output))
if __name__ == "__main__":
    app.run(debug=True)
