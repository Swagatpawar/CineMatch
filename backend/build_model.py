import pickle
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import os
    import sys

    sys.path.insert(0, os.path.dirname(__file__))

from cinematch_model import CineMatchSVD

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
MODEL_PATH = BASE / "models" / "cinematch_svd_final.pkl"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    movies = pd.read_csv(DATA_DIR / "movies.csv")
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")

    if "movieId" in movies.columns:
        movies = movies.rename(columns={"movieId": "movie_id"})
    if "userId" in ratings.columns:
        ratings = ratings.rename(columns={"userId": "user_id", "movieId": "movie_id"})

    movies["movie_id"] = movies["movie_id"].astype(int)
    ratings["user_id"] = ratings["user_id"].astype(int)
    ratings["movie_id"] = ratings["movie_id"].astype(int)
    ratings["rating"] = ratings["rating"].astype(float)
    return movies, ratings


def main() -> None:
    movies, ratings = load_data()
    user_ids = sorted(ratings["user_id"].unique())
    movie_ids = sorted(movies["movie_id"].unique())
    user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    movie_to_idx = {mid: idx for idx, mid in enumerate(movie_ids)}

    matrix = np.zeros((len(user_ids), len(movie_ids)), dtype=float)
    for row in ratings[["user_id", "movie_id", "rating"]].itertuples(index=False):
        matrix[user_to_idx[int(row[0])], movie_to_idx[int(row[1])]] = float(row[2])

    mask = matrix > 0
    rating_values = matrix[mask]
    global_mean = float(rating_values.mean()) if rating_values.size else 0.0

    user_means = np.zeros(len(user_ids), dtype=float)
    for idx in range(len(user_ids)):
        values = matrix[idx, mask[idx]] if mask[idx].any() else np.array([])
        user_means[idx] = float(values.mean()) if values.size else global_mean

    movie_means = np.zeros(len(movie_ids), dtype=float)
    for idx in range(len(movie_ids)):
        values = matrix[mask[:, idx], idx] if mask[:, idx].any() else np.array([])
        movie_means[idx] = float(values.mean()) if values.size else global_mean

    centered = np.zeros_like(matrix)
    for i in range(len(user_ids)):
        for j in range(len(movie_ids)):
            if mask[i, j]:
                centered[i, j] = matrix[i, j] - user_means[i] - movie_means[j] + global_mean

    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    latent = min(50, len(s))
    u_k = u[:, :latent] * np.sqrt(s[:latent])
    v_k = vt[:latent, :] * np.sqrt(s[:latent])[:, None]

    model = CineMatchSVD(user_ids, movie_ids, u_k, v_k.T, user_means, movie_means, global_mean)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as handle:
        pickle.dump(model, handle)

    print(f"Saved model to {MODEL_PATH}")
    print(f"Users={len(user_ids)} Movies={len(movie_ids)} Factors={latent}")
    print(f"Sample prediction for user 1 / movie 1 => {model.predict(1, 1).est:.4f}")


if __name__ == "__main__":
    main()
