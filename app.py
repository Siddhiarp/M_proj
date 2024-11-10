from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('naive_bayes_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the review from the form
    review = request.form['review']
    
    # Transform the review using the vectorizer
    review_tfidf = vectorizer.transform([review])
    
    # Predict the class
    prediction = model.predict(review_tfidf)
    
    # Map the prediction to the class name
    class_name = 'recommended' if prediction[0] == 1 else 'not recommended'
    
    return render_template('index.html', prediction_text=f'This game is {class_name}')

if __name__ == "__main__":
    app.run(debug=True)
