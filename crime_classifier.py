import joblib

# Load trained model
model = joblib.load("crime_classifier.pkl")

def classify_crime(text):
    prediction = model.predict([text])[0]
    return {"label": prediction}

# Example usage
if __name__ == "__main__":
    sample_text = "Someone hacked my social media account and is demanding money."
    print(classify_crime(sample_text))
