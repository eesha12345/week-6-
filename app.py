import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Streamlit page
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="😊"
)

st.title("😊 Sentiment Analysis Classifier")
st.write("Enter a review below to predict whether it is Positive or Negative.")

review = st.text_area("Enter your review")

if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        # 1. Clear punctuation and check text directly
        clean_review = review.lower().replace("'", "").replace('"', "")
        words = clean_review.split()

        # 2. Extract features and predict using the model
        review_vector = vectorizer.transform([review])
        prediction = model.predict(review_vector)
        pred_string = str(prediction).lower()

        # 3. Direct matching rules to bypass model bias completely
        neg_words = ["dont", "not", "bad", "terrible", "worst", "hate", "no", "disappointed", "poor"]
        pos_words = ["like", "good", "love", "great", "excellent", "awesome", "amazing", "satisfied"]

        # Check if any negative or positive words are in the text
        has_neg = any(w in words for w in neg_words) or any(w in clean_review for w in ["dont ", "not "])
        has_pos = any(w in words for w in pos_words)

        # Final decision gate
        if has_neg:
            is_positive = False
        elif has_pos or "positive" in pred_string or "1" in pred_string:
            is_positive = True
        else:
            is_positive = False

        # 4. Display the output results UI layouts
        if is_positive:
            st.success("😊 Positive Review")
            st.info("💬 **Comment:** Good feedback! The customer is fully satisfied.")
        else:
            st.error("😡 Negative Review")
            st.info("💬 **Comment:** Bad feedback! This review requires attention or manual follow-up.")





