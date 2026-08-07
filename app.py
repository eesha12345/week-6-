import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Streamlit page
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="😊"
)

st.title("😊 Sentiment Analysis Classifier")
st.write("Enter a review below to predict whether it is Positive or Negative.")

review = st.text_area("Enter your review")

if st.button("Predict"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        review_vector = vectorizer.transform([review])
        prediction = model.predict(review_vector)[0]

        if prediction == "positive":
            st.success("😊 Positive Review")
        else:
            st.error("😞 Negative Review")