from flask import Flask, request, jsonify, render_template
import pandas as pd

import joblib
from utils import preprocessor

app = Flask(__name__)
model = joblib.load('model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    process_input = request.form['text']
    predicted_sentiment = model.predict(pd.Series(process_input))[0]
    if predicted_sentiment == 1:
        output = 'positive'
    else:
        output = 'negative'

    return render_template('index.html', sentiment=f'Predicted sentiment of "{process_input}" is {output}.')


if __name__ == "__main__":
    app.run(debug=True)
