import pandas as pd
import joblib

from preprocess import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("dataset.csv")

# Clean reviews
df["review"] = df["review"].apply(clean_text)

# Features and labels
X = df["review"]
y = df["sentiment"]

# Vectorize
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Train on all data
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save model
joblib.dump(model, "models/sentiment_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("Model Saved Successfully!")