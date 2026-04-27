import pandas as pd

fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")
fake["label"] = 1
real["label"] = 0

df = pd.concat([fake, real], ignore_index=True).sample(frac=1, random_state=42)
df["text"] = (df["title"] + " " + df["text"]).str.strip()
df = df[["text", "label"]].dropna()
df.to_csv("news.csv", index=False)
print(f"✅ Done! Saved {len(df):,} rows to news.csv")