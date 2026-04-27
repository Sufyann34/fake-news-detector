import re
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import os
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

# 1. Load data
df = pd.read_csv("news.csv")
print(f"✅ Loaded {len(df):,} rows")

# 2. Better cleaning
def clean(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Remove stopwords
    words = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return " ".join(words)

df["text"] = df["text"].apply(clean)
print("✅ Text cleaned")

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# 4. Better TF-IDF — focuses on meaning not source style
vec = TfidfVectorizer(
    max_features=100000,
    ngram_range=(1, 3),
    sublinear_tf=True,
    min_df=2,
    max_df=0.95,   # ignores words that appear in 95%+ of docs (too common)
)
X_train_tf = vec.fit_transform(X_train)
X_test_tf  = vec.transform(X_test)
print("✅ TF-IDF done")

# 5. Train with better regularization
model = LogisticRegression(max_iter=1000, C=1, solver="lbfgs")
model.fit(X_train_tf, y_train)
print("✅ Model trained")

# 6. Evaluate
preds = model.predict(X_test_tf)
acc = accuracy_score(y_test, preds)
print(f"\n🎯 Accuracy: {acc*100:.1f}%")
print(classification_report(y_test, preds, target_names=["Real", "Fake"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, preds))

# 7. Save
os.makedirs("model", exist_ok=True)
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vec, f)
print("\n💾 Model saved to ./model/")