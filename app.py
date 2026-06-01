from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
import numpy as np
import pickle

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

movies = pd.read_csv("movies.csv")

movie_features = pd.read_csv(
    "movie_index.csv"
)

embeddings = np.load(
    "movie_embeddings.npy"
)

with open("knn.pkl", "rb") as f:
    knn = pickle.load(f)

movie_id_to_index = {
    movie_id: idx
    for idx, movie_id in enumerate(
        movie_features.iloc[:,0]
    )
}

index_to_movie_id = {
    idx: movie_id
    for movie_id, idx in movie_id_to_index.items()
}

@app.get("/")
def home():
    return {
        "status": "running",
        "model": "Movie Recommender"
    }

@app.get("/search")
def search(q:str):

    results = movies[
        movies["title"]
        .str.contains(q,
                      case=False,
                      na=False)
    ].head(10)

    return results[
        ["movieId","title","genres"]
    ].to_dict("records")


@app.get("/recommend")
def recommend(movie:str):

    row = movies[
        movies["title"].str.lower()
        ==
        movie.lower()
    ]

    if len(row) == 0:

        return {
            "error":"Movie not found"
        }

    movie_id = int(
        row.iloc[0]["movieId"]
    )

    idx = movie_id_to_index[movie_id]

    movie_embedding = embeddings[idx]

    distances, indices = knn.kneighbors(
        movie_embedding.reshape(1,-1),
        n_neighbors=11
    )

    recs = []

    for dist, i in zip(
        distances[0][1:],
        indices[0][1:]
    ):

        rec_movie_id = index_to_movie_id[i]

        rec_row = movies[
            movies["movieId"]
            ==
            rec_movie_id
        ].iloc[0]

        title = rec_row["title"]

        year = None

        try:
            year = int(
                title[-5:-1]
            )
        except:
            pass

        recs.append({

            "title": title,

            "year": year,

            "similarity": round(
                float(1-dist),
                3
            ),

            "genres": str(
                rec_row["genres"]
            ).split("|")
        })

    return {
        "recommendations": recs
    }