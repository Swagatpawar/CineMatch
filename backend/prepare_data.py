import pickle
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
MODELS_DIR = BASE / "models"
DATA_DIR.mkdir(exist_ok=True, parents=True)
MODELS_DIR.mkdir(exist_ok=True, parents=True)

ZIP_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
ZIP_PATH = DATA_DIR / "ml-latest-small.zip"

if not ZIP_PATH.exists():
    print("Downloading MovieLens dataset...")
    urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)

if not (DATA_DIR / "ml-latest-small" / "movies.csv").exists():
    with zipfile.ZipFile(ZIP_PATH) as archive:
        archive.extractall(DATA_DIR)

source_dir = DATA_DIR / "ml-latest-small"
movies_path = source_dir / "movies.csv"
ratings_path = source_dir / "ratings.csv"

if movies_path.exists() and not (DATA_DIR / "movies.csv").exists():
    shutil.copy2(movies_path, DATA_DIR / "movies.csv")

if ratings_path.exists() and not (DATA_DIR / "ratings.csv").exists():
    shutil.copy2(ratings_path, DATA_DIR / "ratings.csv")

movies_df = pd.read_csv(DATA_DIR / "movies.csv")
ratings_df = pd.read_csv(DATA_DIR / "ratings.csv")
movies_df = movies_df[["movieId", "title", "genres"]].rename(columns={"movieId": "movie_id"})
ratings_df = ratings_df[["userId", "movieId", "rating"]].rename(columns={"userId": "user_id", "movieId": "movie_id"})
movies_df.to_csv(DATA_DIR / "movies.csv", index=False)
ratings_df.to_csv(DATA_DIR / "ratings.csv", index=False)

# Real latent-factor SVD trained from the MovieLens dataset.
user_ids = sorted(ratings_df["user_id"].unique())
movie_ids = sorted(movies_df["movie_id"].unique())
user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}

matrix = np.zeros((len(user_ids), len(movie_ids)), dtype=float)
for row in ratings_df.itertuples(index=False):
    matrix[user_to_idx[row.user_id], movie_to_idx[row.movie_id]] = row.rating

global_mean = float(np.nanmean(matrix[matrix > 0])) if np.any(matrix > 0) else 0.0
mask = matrix > 0
row_means = np.zeros(len(user_ids), dtype=float)
for i in range(len(user_ids)):
    values = matrix[i, mask[i]] if mask[i].any() else np.array([])
    row_means[i] = values.mean() if values.size else global_mean

col_means = np.zeros(len(movie_ids), dtype=float)
for j in range(len(movie_ids)):
    values = matrix[mask[:, j], j] if mask[:, j].any() else np.array([])
    col_means[j] = values.mean() if values.size else global_mean

centered = np.where(mask, matrix - row_means[:, None] - col_means[None, :] + global_mean, 0.0)
U, S, Vt = np.linalg.svd(centered, full_matrices=False)
latent = 50
U_k = U[:, :latent] * np.sqrt(S[:latent])
V_k = Vt[:latent, :] * np.sqrt(S[:latent])[:, None]

class Prediction:
    def __init__(self, uid, iid, est):
        self.uid = uid
        self.iid = iid
        self.est = est

class CineMatchSVD:
    def __init__(self, user_ids, movie_ids, user_factors, movie_factors, user_bias, movie_bias, global_mean):
        self.user_ids = user_ids
        self.movie_ids = movie_ids
        self.user_factors = user_factors
        self.movie_factors = movie_factors
        self.user_bias = user_bias
        self.movie_bias = movie_bias
        self.global_mean = global_mean
        self.n_factors = user_factors.shape[1]

    def predict(self, uid, iid):
        if uid not in self.user_ids:
            base = self.global_mean
        else:
            u_idx = self.user_ids.index(uid)
            u_bias = self.user_bias[u_idx]
            if iid not in self.movie_ids:
                return Prediction(uid, iid, self.global_mean + u_bias)
            i_idx = self.movie_ids.index(iid)
            score = self.global_mean + u_bias + self.movie_bias[i_idx] + np.dot(self.user_factors[u_idx], self.movie_factors[i_idx])
            return Prediction(uid, iid, float(np.clip(score, 0.5, 5.0)))
        return Prediction(uid, iid, float(np.clip(base, 0.5, 5.0)))

model = CineMatchSVD(
    list(user_ids),
    list(movie_ids),
    U_k,
    V_k.T,
    row_means,
    col_means,
    global_mean,
)

with (MODELS_DIR / "cinematch_svd_final.pkl").open("wb") as handle:
    pickle.dump(model, handle)

print(f"Model saved: {MODELS_DIR / 'cinematch_svd_final.pkl'}")
print(f"Users: {len(user_ids)} | Movies: {len(movie_ids)} | Latent factors: {latent}")
print(f"Global mean: {global_mean:.4f}")
print("Sample prediction:", model.predict(1, 1).est)
