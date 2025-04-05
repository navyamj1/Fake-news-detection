import pandas as pd

# Load datasets
fake_df = pd.read_csv("archive/Fake.csv")
true_df = pd.read_csv("archive/True.csv")

# Add labels
fake_df["label"] = "FAKE"
true_df["label"] = "REAL"

# Combine and shuffle
data = pd.concat([fake_df, true_df])
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to news.csv
data.to_csv("news.csv", index=False)

print("news.csv created successfully.")
