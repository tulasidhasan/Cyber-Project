import pandas as pd
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Download required nltk data
nltk.download("punkt")

# Initialize the Porter Stemmer
stemmer = PorterStemmer()

# Function for stemming and tokenization
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Convert to lowercase and tokenize
    stemmed_tokens = [stemmer.stem(word) for word in tokens]  # Apply stemming
    return " ".join(stemmed_tokens)  # Join back into a string

# Load dataset
df = pd.read_csv("cybercrime_updated_dataset.csv")

# Check for missing values
df.dropna(inplace=True)

# Apply preprocessing
df["processed_text"] = df["text"].apply(preprocess_text)

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(df["processed_text"], df["crime_type"], test_size=0.2, random_state=42)

# Create pipeline (TF-IDF with custom tokenizer + Naive Bayes)
vectorizer = TfidfVectorizer(tokenizer=word_tokenize, preprocessor=preprocess_text)
model = make_pipeline(vectorizer, MultinomialNB())

# Train model
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "crime_classifier.pkl")
print("Model saved as crime_classifier.pkl")
