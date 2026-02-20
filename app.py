import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from rapidfuzz import process
import requests

# -------------------------
# Load Data
# -------------------------
@st.cache_data
def load_data():
    ratings = pd.read_csv(
        "data/u.data",
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"]
    )
    movies = pd.read_csv(
        "data/u.item",
        sep="|",
        encoding="latin-1",
        header=None,
        usecols=[0, 1],
        names=["item_id", "title"]
    )
    df = ratings.merge(movies, on="item_id")
    return df

df = load_data()

# -------------------------
# Movie-User Matrix & KNN
# -------------------------
movie_user_matrix = df.pivot_table(
    index="title",
    columns="user_id",
    values="rating"
).fillna(0)

model = NearestNeighbors(metric="cosine", algorithm="brute")
model.fit(movie_user_matrix.values)

# -------------------------
# Recommendation Function
# -------------------------
def recommend_movies(movie_name, n_recommendations=5):
    if movie_name not in movie_user_matrix.index:
        return []

    movie_vector = movie_user_matrix.loc[movie_name].values.reshape(1, -1)
    distances, indices = model.kneighbors(movie_vector, n_neighbors=n_recommendations + 1)

    recommendations = []
    for i in range(1, len(distances.flatten())):
        recommendations.append(
            (movie_user_matrix.index[indices.flatten()[i]], 1 - distances.flatten()[i])
        )
    return recommendations

# -------------------------
# Poster Fetching (OMDb API)
# -------------------------
def get_poster(movie_title):
    api_key = "e728387a"  # Buraya kendi key’ini koy
    # Parantezli yılı kaldır
    title = movie_title.split(" (")[0]
    url = f"http://www.omdbapi.com/?t={title}&apikey={api_key}"
    
    try:
        response = requests.get(url).json()
        if response.get("Poster") and response["Poster"] != "N/A":
            return response["Poster"]
    except:
        return None
    return None

# -------------------------
# Streamlit UI
# -------------------------
st.title("🎬 Movie Recommendation System ")
st.write("Search for a movie and get personalized recommendations!")

movie_list = movie_user_matrix.index.tolist()
search_term = st.text_input("Type a movie name:", key="movie_input")
selected_movie = None

if search_term:
    # Fuzzy match
    match, score, _ = process.extractOne(search_term, movie_list)
    if score > 60:
        selected_movie = match
        st.write(f"Did you mean: **{match}**? (Confidence: {score:.0f}%)")
    else:
        st.warning("No close match found. Try another title.")

# Recommendations
# Recommendations
if selected_movie:
    recommendations = recommend_movies(selected_movie)

    if recommendations:
        st.subheader("🎥 Recommended Movies:")
        for movie, similarity in recommendations:
            st.write(f"⭐ {movie} — Similarity: {similarity:.2f}")
            poster_url = get_poster(movie)
            if poster_url:
                st.image(poster_url, width=150)
            else:
                st.write("Poster not available")