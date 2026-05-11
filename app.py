import streamlit as st
import pickle
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ─── Page Config ────────────────────────────────────────
st.set_page_config(
    page_title="Music Recommendation System", page_icon="🎵", layout="centered"
)


# ─── Load Model ─────────────────────────────────────────
@st.cache_resource
def load_model():
    tf_idf = pickle.load(open("tfidf.pkl", "rb"))
    df = pickle.load(open("df.pkl", "rb"))
    indices = pickle.load(open("indices.pkl", "rb"))
    tfidf_matrix = tf_idf.transform(df["clean_text"])
    return tf_idf, df, indices, tfidf_matrix


tf_idf, df, indices, tfidf_matrix = load_model()

# ─── Preprocessing (same as training) ───────────────────
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_song_name(name):
    name = re.sub(r"[''\"(),:;!?]", "", str(name))
    name = re.sub(r"\s+", " ", name).strip()
    return name


# Map display name → original name
df["display_name"] = df["song"].apply(clean_song_name)
song_map = dict(zip(df["display_name"], df["song"]))


# ─── Recommendation Function ────────────────────────────
def recommend_songs(song_name, n=10):
    if song_name not in indices:
        return None

    idx = indices[song_name]

    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    sim_indices = sim_scores.argsort()[::-1][1 : n + 1]

    result = df[["song", "artist"]].iloc[sim_indices].copy()
    result["similarity_score"] = sim_scores[sim_indices].round(4)
    result = result.reset_index(drop=True)
    result.index += 1

    return result


# ─── UI ─────────────────────────────────────────────────
st.title("Music Recommendation System")
st.markdown("Find songs similar to your favorite music based on **lyrics analysis**!")
st.divider()

# Song Selection
song_list = sorted(df["display_name"].unique().tolist())
selected_display = st.selectbox("Search or select a song:", song_list)
selected_song = song_map[selected_display]

# Artist info
artist = df[df["song"] == selected_song]["artist"].values[0]
st.markdown(f"Artist:{artist}")

st.divider()

# Number of recommendations
n_recommendations = st.slider(
    "How many recommendations?", min_value=5, max_value=20, value=10
)

# ─── Recommend Button ────────────────────────────────────
if st.button("Get Recommendations", use_container_width=True):
    with st.spinner("Finding similar songs..."):
        recommendations = recommend_songs(selected_song, n_recommendations)

    if recommendations is not None:
        st.success(f"Top {n_recommendations} songs similar to **{selected_display}**:")
        st.dataframe(
            recommendations,
            use_container_width=True,
            column_config={
                "song": "Song",
                "artist": "Artist",
                "similarity_score": st.column_config.ProgressColumn(
                    "Similarity",
                    min_value=0,
                    max_value=1,
                ),
            },
        )
    else:
        st.error("Song not found! Please try another song.")

# ─── Footer ─────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center; color:gray;'>Built by Abdelrahman Eldera • Content-Based Filtering using TF-IDF & Cosine Similarity</p>",
    unsafe_allow_html=True,
)
