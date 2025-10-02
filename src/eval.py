import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR = "data/"
RESONTS_DIR = "results/"

def load_and_eval(model_fname):
    preproc, model = joblib.load('results/' + model_fname)
    X_test = pd.read_csv(DATA_DIR + 'X_test.csv')
    y_test = pd.read_csv(DATA_DIR + 'y_test.csv')['label']
    X_test_trans = preproc.transform(X_test)
    y_pred = model.predict(X_test_trans)
    print(classification_report(y_test, y_pred))
    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))

if _name_ == '_main_':
    load_and_eval('RandomForest_model.pkl')  # replace filename as needed