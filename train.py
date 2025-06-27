import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load dataset
data = pd.read_csv("language.csv")  

# Extract features and labels
x = np.array(data['Text'])
y = np.array(data['language'])

# Convert text data to feature vectors
cv = CountVectorizer()
X = cv.fit_transform(x)

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Train model
model = MultinomialNB()
model.fit(x_train, y_train)

# Save model and vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(cv, "vectorizer.pkl")

print("Model and vectorizer saved successfully.")
