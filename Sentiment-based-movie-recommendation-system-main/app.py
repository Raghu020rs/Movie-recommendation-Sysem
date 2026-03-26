import streamlit as st
import pandas as pd
from textblob import TextBlob
import os

# Set page config
st.set_page_config(page_title="🎬 Sentiment-Based Movie Recommender", layout="centered")

# Load data
@st.cache_data
def load_movie_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "sentiment_movies.csv")
    return pd.read_csv(file_path)

df = load_movie_data()

# Sentiment Analysis Function
def analyze_sentiment(review):
    blob = TextBlob(review)
    polarity = blob.sentiment.polarity
    if polarity > 0.2:
        return "positive"
    elif polarity < -0.2:
        return "negative"
    else:
        return "neutral"

# Recommendation Function
def recommend_movies(sentiment_label):
    filtered = df[df['sentiment'] == sentiment_label]
    return filtered[['title', 'genres']].sample(n=min(5, len(filtered)))

# UI
st.title("🎥 Sentiment-Based Movie Recommender System")
st.markdown("Write a short review, and we'll recommend movies based on how you feel!")

user_review = st.text_area("✍️ Write your movie review here...")

if st.button("Analyze and Recommend"):
    if user_review.strip():
        sentiment = analyze_sentiment(user_review)
        st.info(f"Detected Sentiment: **{sentiment.upper()}**")

        recommendations = recommend_movies(sentiment)
        st.success("🎯 Recommended Movies:")
        for idx, row in recommendations.iterrows():
            st.write(f"**{row['title']}** — _{row['genres']}_")
    else:
        st.warning("Please write a review to get recommendations.")
