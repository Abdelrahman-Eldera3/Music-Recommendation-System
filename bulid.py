# Imports
import numpy as np
import pandas as pd
import os
import re
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("punkt_tab")


# Load data
df = pd.read_csv("spotify_millsongdata.csv")

# Sample 10,000 songs to avoid memory issues
df = df.sample(10000, random_state=42).reset_index(drop=True)

print("✅ Data loaded!")
print("Shape:", df.shape)
df.head()


print("Columns    :", df.columns.tolist())
print("Null values:\n", df.isnull().sum())
print("Duplicated :", df.duplicated().sum())
df.info()

# EDA — Top Artists
plt.figure(figsize=(12, 5))
top_artist = df["artist"].value_counts().head(15)
sns.barplot(x=top_artist.values, y=top_artist.index, palette="viridis")
plt.title("Top 15 Artists by Number of Songs")
plt.xlabel("Number of Songs")
plt.tight_layout()
plt.savefig("top_artists.png", dpi=150)
plt.show()


# EDA — Lyrics Length Distribution
df["text_length"] = df["text"].apply(lambda x: len(str(x).split()))

plt.figure(figsize=(10, 4))
sns.histplot(df["text_length"], bins=50, color="steelblue")
plt.title("Distribution of Lyrics Word Count")
plt.xlabel("Word Count")
plt.tight_layout()
plt.show()

print("Average lyrics length:", df["text_length"].mean().round(1), "words")


# Text Preprocessing
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    # 1. Lowercasing
    text = str(text).lower()

    # 2. Remove special characters & numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # 3. Tokenization
    tokens = word_tokenize(text)

    # 4. Remove Stopwords + Lemmatization
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 2
    ]

    return " ".join(tokens)


df["clean_text"] = df["text"].apply(clean_text)

# Verify
print("Before:", df["text"][0][:200])
print("\nAfter:", df["clean_text"][0][:200])


# TF-IDF Vectorization
tf_idf = TfidfVectorizer(max_features=5000)
tfidf_matrix = tf_idf.fit_transform(df["clean_text"])

print("TF-IDF Matrix shape:", tfidf_matrix.shape)


# Song Index Mapping
indices = pd.Series(df.index, index=df["song"]).drop_duplicates()
print("Total unique songs indexed:", len(indices))


# Recommendation Function
def recommend_songs(song_name, n=10):
    # Check if song exists
    if song_name not in indices:
        print(f" Song '{song_name}' not found in dataset!")
        return None

    # Get song index
    idx = indices[song_name]

    # Compute similarity on-demand (memory efficient)
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    # Get top N (exclude the song itself)
    sim_indices = sim_scores.argsort()[::-1][1 : n + 1]

    # Build result dataframe
    result = df[["song", "artist"]].iloc[sim_indices].copy()
    result["similarity_score"] = sim_scores[sim_indices].round(4)
    result = result.reset_index(drop=True)
    result.index += 1

    return result


#  Test Model
song = df["song"].iloc[0]
print(f"🎵 Testing recommendations for: '{song}'\n")
recommendations = recommend_songs(song, n=10)
print(recommendations)


# Save Model ─────────────────────────────────────────────────
pickle.dump(tf_idf, open("tfidf.pkl", "wb"))
pickle.dump(df[["song", "artist", "clean_text"]], open("df.pkl", "wb"))
pickle.dump(indices, open("indices.pkl", "wb"))

print("Model saved successfully!")
print("tfidf.pkl")
print("df.pkl(with clean_text)")
print("indices.pkl")
