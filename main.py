# To run the Streamlit app, use the 'streamlit run main.py' command in the terminal

# Import necessary libraries
import streamlit as st
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load pre-trained model and TF-IDF vectorizer
with open('nb.pkl', 'rb') as f:
    nb = pickle.load(f)
with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Function to clean text data
def clean_text(text):
    # Convert text to lowercase
    text = str(text).lower()
    # Remove square brackets and content inside them
    text = re.sub('\[.*?\]', '', text)
    # Remove URLs
    text = re.sub('https?://\S+|www\.\S+', '', text)
    # Remove HTML tags
    text = re.sub('<.*?>+', '', text)
    # Remove punctuation
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    # Remove newline characters
    text = re.sub('\n', '', text)
    # Remove words containing digits
    text = re.sub('\w*\d\w*', '', text)
    return text

# Function to preprocess text data
def preprocess_data(text):
    # Remove stopwords
    stopwords_set = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stopwords_set)
    # Lemmatize words
    lemma = WordNetLemmatizer()
    text = ' '.join(lemma.lemmatize(word) for word in text.split())
    # Remove short words (length less than 3)
    text = ' '.join(word for word in text.split() if len(word) > 2)
    return text

# Function to predict sentiment of a comment
def predict_sentiment(comment, model, tfidf):
    # Convert comment into a list
    comment_list = [comment]
    # Transform comment into TF-IDF vector
    comment_vector = tfidf.transform(comment_list)
    # Predict sentiment using the trained model
    comment_prediction = model.predict(comment_vector)[0]
    # Return the sentiment prediction
    if comment_prediction == 1:
        return "Positive comment"
    else:
        return "Negative comment"

# Function to define Streamlit app
def main():
    # Set title of the Streamlit app
    st.title('Sentiment Analysis')

    # Create input box for user's comment
    user_comment = st.text_area('Enter your comment here:', '')

    # Button to predict sentiment
    if st.button('Predict Sentiment'):
        # Check if user entered a comment
        if user_comment.strip() == '':
            st.error('Please enter a comment.')
        else:
            # Preprocess the user's comment
            preprocess_comment = preprocess_data(user_comment)
            # Predict sentiment
            sentiment_prediction = predict_sentiment(preprocess_comment, nb, tfidf)
            # Display prediction
            st.write('Prediction:', sentiment_prediction)

# Execute the Streamlit app
if __name__ == '__main__':
    main()

