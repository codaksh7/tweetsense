import joblib
import spacy
import math
from textblob import TextBlob
from transformers import pipeline
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_cors import CORS # You wrote flaskcors (missing underscore) in your earlier description
import numpy as np
# --- MODEL LOADING & UTILITIES ---
try:
    vectorizer = joblib.load('vectorizer.sav')
    model = joblib.load('trained_model.sav')
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    priority_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    print("Models and NLP pipeline loaded successfully.")
except Exception as e:
    print(f"Error loading models or Spacy: {e}")

def preprocess_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

def prioritize_comment(text):
    candidate_labels = ["High Priority", "Medium Priority", "Low Priority"]
    try:
        result = priority_classifier(text, candidate_labels)
        return result['labels'][0]
    except:
        return "Not Classified"

def get_sentiment_label(text, prediction):
    polarity = TextBlob(text).sentiment.polarity
    if -0.1 <= polarity <= 0.1:
        return "Neutral"
    else:
        return "Negative" if prediction[0] == 0 else "Positive"

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return "Backend is running. Use POST /api/analyze for sentiment analysis.", 200

@app.route('/api/model_info', methods=['GET'])
def model_info():
    try:
        vectorizer_vocab_size = len(vectorizer.vocabulary_)
        model_classes = model.classes_.tolist()
        sample_vocab = list(vectorizer.vocabulary_.keys())[:10]
        return jsonify({
            "status": "success",
            "vectorizer_vocab_size": vectorizer_vocab_size,
            "sample_vocab": sample_vocab,
            "model_classes": model_classes
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to get model info: {str(e)}"
        }), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_sentiment():
    try:
        data = request.json
        user_input = data.get('text', '')

        if not user_input or len(user_input.split()) < 3:
            return jsonify({
                "error": "Invalid input.",
                "message": "Please submit text with at least 3 words for analysis."
            }), 400

        processed_text = preprocess_text(user_input)
        X = vectorizer.transform([processed_text])
        prediction = model.predict(X)
        proba = model.predict_proba(X)[0]
        confidence = max(proba)
        sentiment_label = get_sentiment_label(user_input, prediction)
        priority = prioritize_comment(user_input)

        return jsonify({
            "status": "success",
            "sentiment": sentiment_label,
            "confidence": f"{confidence:.4f}",
            "priority": priority,
            "input_length": len(user_input.split())
        }), 200

    except Exception as e:
        print(f"Analysis error: {e}")
        return jsonify({
            "status": "error",
            "message": f"An internal error occurred: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
