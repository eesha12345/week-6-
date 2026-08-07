import joblib
from preprocess import clean_text

# Load model and vectorizer
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Take input
review = input("Enter a review: ")

# Clean review
cleaned_review = clean_text(review)

# Convert to vector
review_vector = vectorizer.transform([cleaned_review])

# Predict
prediction = model.predict(review_vector)[0]

print("Prediction:", prediction)