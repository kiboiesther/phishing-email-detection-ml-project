import os

# Tell Kaggle where to find kaggle.json (project-based setup)
os.environ['KAGGLE_CONFIG_DIR'] = os.path.join(os.getcwd(), ".kaggle")

# Create data folder if not exists
if not os.path.exists("data"):
    os.makedirs("data")

# Kaggle dataset slug
dataset_slug = "subhajournal/phishingemails"

if not os.path.exists("data/phishing.csv"):
    print("Downloading dataset from Kaggle...")
    os.system(f"kaggle datasets download -d {dataset_slug} -p data --unzip")
    print("Download complete!")
else:
    print("Dataset already exists. Skipping download.")

