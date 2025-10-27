🟣 TweetSense Analyzer: NLP-Powered Sentiment Classification

🌟 Project Overview

TweetSense Analyzer is a foundational Natural Language Processing (NLP) mini-project designed to classify the sentiment of social media text (simulating Twitter data). This application demonstrates a complete machine learning pipeline, from model training and vectorization to deployment via a user-friendly Streamlit web interface.

The core goal is to accurately categorize text inputs as Positive, Negative, or Neutral and provide a visual analysis of sentiment distribution and key themes in batch data.

🛠️ Technology Stack

This project leverages a hybrid stack for robust performance:

Component

Technology

Role

Frontend/App

Streamlit

Creates the fast, interactive, and beautifully styled web interface.

Machine Learning

Scikit-learn

Provides the Logistic Regression model for classification.

NLP Pipeline

SpaCy

Used for efficient Text Preprocessing and Lemmatization.

Text Analysis

TextBlob

Used for polarity calculation to establish the Neutrality threshold.

Summarization

Sumy

Used for fast, abstractive Summary Generation (LSA Summarizer).

Data Handling

Joblib, Pandas

Used for serializing/loading the model and managing batch data processing.

🧠 Core NLP Pipeline & Classification Rules

The robustness of TweetSense lies in its carefully constructed NLP workflow:

1. Text Preprocessing (spacy and joblib):

Before classification, all user input undergoes strict preprocessing to ensure model compatibility:

Lemmatization: The SpaCy library is used to reduce words to their base or dictionary form (e.g., "running," "ran," "runs" $\rightarrow$ "run"). This is crucial because the model's vocabulary (vectorizer.sav) is based on lemmatized words.

Vectorization: The pre-trained TF-IDF Vectorizer (vectorizer.sav) transforms the clean text into a numerical feature matrix suitable for the Logistic Regression model.

2. Sentiment Classification Logic:

The final sentiment label is determined by a hybrid rule set defined in the Python code:

Scenario

Rule

Output

Neutral Range

Polarity Score (from TextBlob) falls between -0.1 and +0.1.

Neutral

Positive/Negative

Polarity is outside the Neutral range, and the Logistic Regression Model predicts 1.

Positive

Positive/Negative

Polarity is outside the Neutral range, and the Logistic Regression Model predicts 0.

Negative

This hybrid approach ensures that tweets that are truly ambiguous or lack strong emotional markers are correctly labeled "Neutral," providing higher accuracy than simple binary classification.

✨ Features

1. Single Text Analysis

Instant Sentiment: Provides immediate classification (Positive, Negative, Neutral) and the model's confidence score.

Summary Generation: Automatically generates a concise summary of the input text using the lightweight Sumy (LSA Summarizer).

2. Batch CSV Analysis (Key Feature)

Bulk Processing: Accepts CSV files (must contain a text column) and runs all entries through the full NLP pipeline.

Visual Distribution: Generates Plotly Pie Charts and Bar Charts showing the overall percentage breakdown of Positive, Negative, and Neutral comments in the uploaded dataset.

Keyword Analysis: Creates a visual WordCloud to quickly identify the most frequently used keywords and themes across all comments.

Downloadable Results: Exports the complete analysis (including original text, predicted sentiment, and confidence) back to a downloadable CSV file.

🚀 Getting Started

Follow these steps to clone and run the application locally.

Prerequisites

Python 3.x installed.

Git installed.

Required Files: Ensure the following files are present in your backend/ folder:

trained_model.sav

vectorizer.sav

Installation Steps

Clone the Repository:

git clone [YOUR_REPOSITORY_URL]
cd TWITTER-SENTIMENT-ANALYSIS/backend

Create and Activate Virtual Environment (venv):

python -m venv venv
.\venv\Scripts\Activate # On Windows PowerShell

# source venv/bin/activate # On Mac/Linux

Install Python Dependencies:

pip install streamlit scikit-learn joblib spacy textblob sumy wordcloud numpy plotly

Download SpaCy Model: This is required for text preprocessing:

python -m spacy download en_core_web_sm

Running the Application

After installation, start the Streamlit application from the backend folder:

streamlit run app2.py
