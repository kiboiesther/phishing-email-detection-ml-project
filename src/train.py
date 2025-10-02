import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURE_PATH = "data/feature_engineered_emails.csv"
RESULTS_DIR = "results/"

def main():
    df = pd.read_csv(FEATURE_PATH)
    df['cleaned_text'] = df['cleaned_text'].fillna('')
    X = df.drop(columns=['label'])
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print('Train/test ready. Add preprocessing and training steps here.')

if _name_ == '_main_':
    main()