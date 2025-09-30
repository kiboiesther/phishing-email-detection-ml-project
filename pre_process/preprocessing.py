import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from scipy.sparse import hstack
import os

# -------------------------
# 1. Load already cleaned dataset from Esther
# -------------------------
df = pd.read_csv("data/cleaned_phishing_emails.csv")
print("✅ Loaded cleaned dataset:", df.shape)
print(df.head())

# Ensure cleaned_text is always a string
df['cleaned_text'] = df['cleaned_text'].fillna("").astype(str)

# -------------------------
# 2. Convert Labels (Safe Email -> 0, Phishing Email -> 1)
# -------------------------
df['label'] = df['Label'].map({
    "Safe Email": 0,
    "Phishing Email": 1
})
print("✅ Converted labels to numeric (0 = Safe, 1 = Phishing)")

# -------------------------
# 3. Handcrafted features
# -------------------------
df['email_length'] = df['cleaned_text'].apply(len)
df['num_digits'] = df['cleaned_text'].apply(lambda x: sum(c.isdigit() for c in x))
df['num_links'] = df['cleaned_text'].apply(lambda x: len(re.findall(r'http[s]?://', str(x))))
df['has_urgent'] = df['cleaned_text'].apply(lambda x: 1 if 'urgent' in x else 0)

# -------------------------
# 4. TF-IDF Features
# -------------------------
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_tfidf = vectorizer.fit_transform(df['cleaned_text'])

# Combine handcrafted + TF-IDF
numeric_features = df[['email_length','num_digits','num_links','has_urgent']].values
X = hstack([X_tfidf, numeric_features])
y = df['label']

# -------------------------
# 5. Feature Selection
# -------------------------
selector = SelectKBest(chi2, k=1000)   # select top 1000 features
X_selected = selector.fit_transform(X, y)

# -------------------------
# 6. Save dataset with engineered features
# -------------------------
os.makedirs("pre_process/data", exist_ok=True)
df[['cleaned_text','label','email_length','num_digits','num_links','has_urgent']].to_csv(
    "pre_process/data/feature_engineered_emails.csv", index=False
)

print("✅ Feature-engineered dataset saved to pre_process/data/feature_engineered_emails.csv")
print("📊 Feature matrix shape (after selection):", X_selected.shape)
print("\nClass distribution:")
print(df['label'].value_counts())
