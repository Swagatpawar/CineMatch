from __future__ import annotations

import pandas as pd

from config import MOVIES_PATH, RATINGS_PATH


class MovieService:
    def __init__(self) -> None:
        self.movies_df: pd.DataFrame | None = None
        self.ratings_df: pd.DataFrame | None = None

    def load_movies(self) -> pd.DataFrame:
        if self.movies_df is not None:
            return self.movies_df

        if not MOVIES_PATH.exists():
            raise FileNotFoundError("movies.csv is missing.")

        self.movies_df = pd.read_csv(MOVIES_PATH)
        if "movie_id" not in self.movies_df.columns and "movieId" in self.movies_df.columns:
            self.movies_df = self.movies_df.rename(columns={"movieId": "movie_id"})
        if "movie_id" not in self.movies_df.columns or "title" not in self.movies_df.columns:
            raise ValueError("movies.csv must include movie_id and title columns.")
        if "genres" not in self.movies_df.columns:
            self.movies_df["genres"] = ""
        self.movies_df["movie_id"] = self.movies_df["movie_id"].astype(int)
        self.movies_df["genres"] = self.movies_df["genres"].fillna("")
        self.movies_df["genres_list"] = self.movies_df["genres"].apply(
            lambda value: [genre.strip() for genre in str(value).split("|") if genre.strip()]
        )
        return self.movies_df

    def load_ratings(self) -> pd.DataFrame:
        if self.ratings_df is not None:
            return self.ratings_df

        if not RATINGS_PATH.exists():
            raise FileNotFoundError("ratings.csv is missing.")

        self.ratings_df = pd.read_csv(RATINGS_PATH)
        if "user_id" not in self.ratings_df.columns and "userId" in self.ratings_df.columns:
            self.ratings_df = self.ratings_df.rename(columns={"userId": "user_id", "movieId": "movie_id"})
        required = {"user_id", "movie_id", "rating"}
        missing = required - set(self.ratings_df.columns)
        if missing:
            raise ValueError(f"ratings.csv missing required columns: {sorted(missing)}")
        self.ratings_df["user_id"] = self.ratings_df["user_id"].astype(int)
        self.ratings_df["movie_id"] = self.ratings_df["movie_id"].astype(int)
        return self.ratings_df

    def get_genres(self) -> list[str]:
        df = self.load_movies()
        genres = sorted({genre for genres in df["genres_list"].tolist() for genre in genres})
        return genres

    def get_movie_lookup(self) -> pd.DataFrame:
        df = self.load_movies().copy()
        df["title_lower"] = df["title"].str.lower()
        return df


movie_service = MovieService()
