# Phishing Email Detection — Project

## Layout
- data/: cleaned & feature-engineered dataset, train/test splits
- notebooks/: Colab-ready notebooks (training, explainability)
- results/: trained models (.pkl), plots (.png), reports
- src/: runnable scripts (train.py, eval.py)
- requirements.txt: python dependencies

## Quick start (Colab)
1. Mount Drive:
   from google.colab import drive
   drive.mount('/content/drive')

2. Open notebooks/train.ipynb in Colab and run cells to reproduce training.
3. Evaluation: use notebooks/explainability.ipynb or run src/eval.py.
