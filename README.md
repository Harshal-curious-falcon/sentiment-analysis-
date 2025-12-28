Sentiment Analysis Project

Overview

This project implements a complete pipeline for sentiment analysis on text data. It covers data loading, preprocessing, exploratory data analysis (EDA), feature extraction using TF-IDF, training multiple machine learning models (Logistic Regression, Naive Bayes, Random Forest), evaluating their performance, and finally, demonstrates how to use the trained model for real-time predictions within a Flask web application.

Features

Data Loading and Exploration: Loads a CSV dataset and performs initial data exploration, including checking for missing values and visualizing sentiment distribution.
Text Preprocessing: Cleans raw text data by converting to lowercase, removing URLs, mentions, hashtags, special characters, numbers, and extra whitespace. It also performs stop word removal and lemmatization.
Exploratory Data Analysis (EDA): Analyzes text length, word counts, and identifies the most common words for each sentiment category, visualizing these insights through histograms and word clouds.
Feature Extraction: Converts preprocessed text into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization.
Model Training: Trains and compares three popular classification models:
Logistic Regression
Multinomial Naive Bayes
Random Forest Classifier
Model Evaluation: Evaluates models based on accuracy, classification reports, and confusion matrices to identify the best-performing model.
Model Persistence: Saves the best-trained model and the TF-IDF vectorizer using Python's pickle module for future use.
Flask Application: Provides a basic Flask application setup to load the saved model and vectorizer, allowing real-time sentiment prediction via a web API endpoint.
Dataset

The project uses the reddit_artist_posts_sentiment.csv dataset, which contains text posts and their corresponding sentiment labels (e.g., 'positive', 'negative', 'neutral').

Project Pipeline (Colab Notebook)

The sentiment analysis pipeline is executed in a sequential manner within the Jupyter/Colab notebook:

Import Libraries: Essential libraries like pandas, numpy, matplotlib, seaborn, wordcloud, nltk, and sklearn components are imported.
NLTK Downloads: Necessary NLTK data packages (punkt, stopwords, wordnet, omw-1.4, punkt_tab) are downloaded.
load_data(file_path): Loads the dataset from the specified CSV file, prints its shape, columns, and the first few rows.
explore_data(df, text_column, sentiment_column): Checks for missing values, displays sentiment distribution, and visualizes it using bar and pie charts.
preprocess_data(df, text_column): Cleans and processes the text data, creating 'cleaned_text' and 'processed_text' columns. This involves steps like lowercasing, removing noise, stop words, and lemmatization.
analyze_sentiments(df, sentiment_column): Performs further EDA by analyzing text length and word counts per sentiment, and identifies most common words. This also includes create_wordclouds(df, sentiment_column) for visual representation of frequent terms.
extract_features(df, sentiment_column): Splits the data into training and testing sets and applies TF-IDF vectorization to convert text into numerical features.
train_models(X_train, X_test, y_train, y_test): Trains Logistic Regression, Multinomial Naive Bayes, and Random Forest classifiers on the extracted features.
evaluate_models(results, y_test): Compares the performance of the trained models, identifies the best one, and displays a detailed classification report and confusion matrix.
sentiment_analysis_pipeline(file_path, text_column, sentiment_column): Orchestrates all the above steps into a single, executable pipeline function.
predict_sentiment(text, model, vectorizer): A utility function to predict the sentiment of new, unseen text using the best-trained model and vectorizer.
Saving Model & Vectorizer: The best model and the TF-IDF vectorizer are saved as sentiment_model.pkl and tfidf_vectorizer.pkl using pickle for deployment.
Flask Application (Deployment)

The app.py file demonstrates how to integrate the trained model into a basic Flask web service. This allows you to send new text via an API endpoint and receive sentiment predictions.

Files Required

app.py: The Flask application code.
sentiment_model.pkl: The pickled machine learning model.
tfidf_vectorizer.pkl: The pickled TF-IDF vectorizer.
Setup and Installation

Create a virtual environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Install dependencies:
pip install Flask scikit-learn numpy pandas nltk
Download NLTK data (if needed in your environment): The app.py includes checks to download necessary NLTK packages if they are not found. However, if you encounter LookupError during startup, you might need to manually run python -m nltk.downloader punkt stopwords wordnet or ensure the app has permissions to download.
Running the Flask App

Save the provided Flask code (from the previous step) as app.py in the same directory as your .pkl files.
Open your terminal, navigate to the directory containing app.py.
Run the Flask application:
python app.py
The app will typically run on http://127.0.0.1:5000/.
Testing the Flask App

You can test the /predict endpoint using curl or a Python script.

Using curl (from your terminal):

curl -X POST -H "Content-Type: application/json" -d '{"text": "This movie is fantastic!"}' http
