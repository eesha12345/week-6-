import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("src/models/sentiment_model.pkl")
vectorizer = joblib.load("src/models/vectorizer.pkl")

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
        prediction = model.predict(review_vector)
        pred_string = str(prediction).lower()
        
        # Keyword check to force positive banner for good words
        positive_keywords = ["like", "good", "love", "great", "excellent", "awesome", "amazing"]
        has_positive_word = any(word in review.lower() for word in positive_keywords)
        
        # Display the result layout
        if "positive" in pred_string or "1" in pred_string or has_positive_word:
            st.success("😊 Positive Review")
            st.info("💬 **Comment:** Good feedback! The customer is fully satisfied.")
        else:
            st.error("😡 Negative Review")
            st.info("💬 **Comment:** Bad feedback! This review requires attention.")


