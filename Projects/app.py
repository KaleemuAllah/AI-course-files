from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

app = Flask(__name__)

# Load the trained model
with open("credit_card_model.pkl", 'rb') as file:
    model = pickle.load(file)

# Load the label encoder
with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    features = [
        request.form['Gender'],
        request.form['Has a car'],
        request.form['Has a property'],
        request.form['Children count'],
        request.form['Employment status'],
        request.form['Education level'],
        request.form['Marital status'],
        request.form['Dwelling'],
        request.form['Age'],
        request.form['Employment length'],
        request.form['Has a mobile phone'],
        request.form['Has a work phone'],
        request.form['Has a phone'],
        request.form['Has an email'],
        request.form['Job title'],
        request.form['Family member count'],
        request.form['Account age'],
        request.form['IncomePerChild']
    ]

    # Transform categorical variables
    features_encoded = []
    for i, val in enumerate(features):
        if i in [0, 1, 2, 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]:
            features_encoded.append(label_encoder[i].transform([val])[0])
        else:
            features_encoded.append(float(val))

    # Make prediction
    prediction = model.predict([features_encoded])[0]

    # Convert prediction to human-readable format
    prediction_label = 'High Risk' if prediction == 1 else 'Low Risk'

    return render_template('index.html', prediction_text='Predicted risk: {}'.format(prediction_label))

if __name__ == "__main__":
    app.run(debug=True)

