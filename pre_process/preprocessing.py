
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources (run once)
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# --- Step 1: Load dataset ---
df = pd.read_csv("data/phishing_Email.csv")
print("Shape before cleaning:", df.shape)

# --- Step 2: Drop missing and duplicates ---
df.dropna(subset=['Email Text', 'Email Type'], inplace=True)
df.drop_duplicates(inplace=True)
print("Shape after cleaning:", df.shape)

# --- Step 3: Define cleaning function ---
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = re.sub(r'<.*?>', ' ', text)  # remove HTML
    text = re.sub(r'http\S+|www\S+', ' ', text)  # remove URLs
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # keep only letters
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# --- Step 4: Apply cleaning ---
df['cleaned_text'] = df['Email Text'].apply(clean_text)

# --- Step 5: Rename target column ---
df.rename(columns={'Email Type': 'Label'}, inplace=True)

import os

# Ensure the directory exists before saving
os.makedirs("pre_process/data", exist_ok=True)

# --- Step 6: Save cleaned dataset ---
df[['cleaned_text', 'Label']].to_csv("pre_process/data/cleaned_phishing_emails.csv", index=False)
print("Cleaned dataset saved to pre_process/data/cleaned_phishing_emails.csv")

# Show preview
print("\nPreview of cleaned dataset:")
print(df[['cleaned_text', 'Label']].head())

# Check class balance
print("\nClass distribution:")
print(df['Label'].value_counts())
