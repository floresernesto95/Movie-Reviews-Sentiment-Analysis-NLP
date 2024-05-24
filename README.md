## Project Description: Movie Reviews Sentiment Analysis

### Overview

The **Movie Reviews Sentiment Analysis** project utilizes Natural Language Processing (NLP) techniques to analyze and determine the sentiment of movie reviews. This project is designed to predict whether a given movie review is positive or negative. The analysis is performed using a pre-trained Naive Bayes model and a TF-IDF vectorizer, integrated into an interactive web application built with Streamlit.

### Key Features

- **Interactive Web Application**: Users can input their movie reviews through a user-friendly interface and receive immediate sentiment predictions.
- **Text Preprocessing**: Reviews are cleaned and preprocessed to remove noise such as punctuation, URLs, HTML tags, and stopwords, ensuring accurate sentiment analysis.
- **Sentiment Prediction**: Utilizes a pre-trained Naive Bayes model to classify the sentiment of reviews as positive or negative.

### Technical Details

1. **Libraries and Tools**:
   - **Streamlit**: For building the web application.
   - **NLTK**: For natural language processing tasks such as lemmatization and stopwords removal.
   - **Scikit-learn**: For model handling and TF-IDF vectorization.
   - **Pickle**: For loading the pre-trained model and vectorizer.

2. **Text Cleaning and Preprocessing**:
   - Convert text to lowercase.
   - Remove unnecessary characters, punctuation, URLs, and HTML tags.
   - Remove stopwords and perform lemmatization to standardize words.
   - Filter out short words to reduce noise.

3. **Model and Vectorizer**:
   - A Naive Bayes classifier trained on a dataset of movie reviews.
   - A TF-IDF vectorizer to convert text data into numerical vectors suitable for model prediction.

4. **Prediction Workflow**:
   - Input review is cleaned and preprocessed.
   - Preprocessed text is transformed into a TF-IDF vector.
   - The model predicts the sentiment based on the vectorized input.
   - Sentiment prediction is displayed to the user.

### How to Run the Application

To run the Streamlit application, use the following command in the terminal:

```bash
streamlit run main.py
```

### Conclusion

The Movie Reviews Sentiment Analysis project showcases the application of NLP and machine learning techniques in sentiment analysis. The interactive web app provides an intuitive interface for users to analyze the sentiment of their movie reviews, making it an excellent tool for movie enthusiasts and researchers alike.
