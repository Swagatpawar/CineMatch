from __future__ import annotations

import math
from typing import Any

import pandas as pd

from config import DEFAULT_RECOMMENDATION_LIMIT, MAX_RECOMMENDATION_LIMIT, POPULAR_MIN_RATINGS
from services.model_service import model_service
from services.movie_service import movie_service


class RecommendationService:
    def get_recommendations_for_user(self, user_id: int, limit: int = DEFAULT_RECOMMENDATION_LIMIT) -> dict:
        if limit < 1:
            limit = DEFAULT_RECOMMENDATION_LIMIT
        if limit > MAX_RECOMMENDATION_LIMIT:
            limit = MAX_RECOMMENDATION_LIMIT

        ratings = movie_service.load_ratings()
        movies = movie_service.load_movies()
        user_ratings = ratings[ratings["user_id"] == user_id]

        if int(user_id) not in ratings["user_id"].unique() or len(user_ratings) < 3:
            popular = self.get_popular_movies(limit)
            return {
                "user_id": user_id,
                "type": "cold_start",
                "message": "Not enough rating history. Showing popular movies.",
                "recommendations": [
                    {
                        "movie_id": row["movie_id"],
                        "title": row["title"],
                        "genres": row["genres_list"],
                        "predicted_rating": float(row["weighted_rating"]),
                    }
                    for _, row in popular.head(limit).iterrows()
                ],
            }

        watched = set(user_ratings["movie_id"].astype(int).tolist())
        model = model_service.get_model()
        candidates = movies[~movies["movie_id"].isin(watched)]
        candidate_ids = candidates["movie_id"].astype(int).tolist()
        predictions = model.predict_many(user_id, candidate_ids)
        recommendation_rows = [
            {
                "movie_id": int(movie["movie_id"]),
                "title": movie["title"],
                "genres": movie["genres_list"],
                "predicted_rating": float(prediction.est),
            }
            for (_, movie), prediction in zip(candidates.iterrows(), predictions)
        ]

        ranked = sorted(recommendation_rows, key=lambda item: item["predicted_rating"], reverse=True)[:limit]
        return {"user_id": user_id, "type": "personalized", "recommendations": ranked}

    def get_popular_movies(self, limit: int = 10) -> pd.DataFrame:
        ratings = movie_service.load_ratings()
        movies = movie_service.load_movies()

        rating_summary = (
            ratings.groupby("movie_id")["rating"]
            .agg(["count", "mean"])
            .reset_index()
            .rename(columns={"count": "rating_count", "mean": "average_rating"})
        )
        rating_summary["weighted_rating"] = (
            (rating_summary["average_rating"] * rating_summary["rating_count"]) 
            / (rating_summary["rating_count"] + POPULAR_MIN_RATINGS)
        )
        merged = rating_summary.merge(movies[["movie_id", "title", "genres", "genres_list"]], on="movie_id", how="left")
        return merged[merged["rating_count"] >= 5].sort_values(["weighted_rating", "rating_count"], ascending=[False, False]).head(limit)

    def get_top_rated_movies(self, limit: int = 10, minimum_ratings: int = 100) -> pd.DataFrame:
        ratings = movie_service.load_ratings()
        movies = movie_service.load_movies()

        rating_summary = (
            ratings.groupby("movie_id")["rating"]
            .agg(["count", "mean"])
            .reset_index()
            .rename(columns={"count": "rating_count", "mean": "average_rating"})
        )
        merged = rating_summary.merge(movies[["movie_id", "title", "genres", "genres_list"]], on="movie_id", how="left")
        return (
            merged[merged["rating_count"] >= minimum_ratings]
            .sort_values(["average_rating", "rating_count"], ascending=[False, False])
            .head(limit)
            .reset_index(drop=True)
        )

    def get_genre_recommendations(self, user_id: int, genre: str, limit: int = DEFAULT_RECOMMENDATION_LIMIT) -> dict:
        movies = movie_service.load_movies()
        ratings = movie_service.load_ratings()
        user_ratings = ratings[ratings["user_id"] == user_id]
        normalized_genre = genre.strip()
        genre_movies = movies[movies["genres_list"].apply(lambda values: normalized_genre in values)]

        if genre_movies.empty:
            raise ValueError(f"Genre '{genre}' does not exist in the dataset.")

        watched = set(user_ratings["movie_id"].astype(int).tolist())
        candidates = genre_movies[~genre_movies["movie_id"].isin(watched)]
        model = model_service.get_model()
        candidate_ids = candidates["movie_id"].astype(int).tolist()
        predictions = model.predict_many(user_id, candidate_ids)
        ranked = [
            {
                "movie_id": int(movie["movie_id"]),
                "title": movie["title"],
                "genres": movie["genres_list"],
                "predicted_rating": float(prediction.est),
            }
            for (_, movie), prediction in zip(candidates.iterrows(), predictions)
        ]

        ranked.sort(key=lambda item: item["predicted_rating"], reverse=True)
        return {
            "user_id": user_id,
            "genre": normalized_genre,
            "recommendations": ranked[:limit],
        }

    def search_movies(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        if not query or not query.strip():
            return []

        movies = movie_service.get_movie_lookup()
        needle = query.strip().lower()
        results = movies[movies["title_lower"].str.contains(needle, case=False, na=False)].head(limit)
        return [
            {
                "movie_id": int(row["movie_id"]),
                "title": row["title"],
                "genres": row["genres_list"],
            }
            for _, row in results.iterrows()
        ]

    def get_user_history(self, user_id: int) -> list[dict[str, Any]]:
        ratings = movie_service.load_ratings()
        movies = movie_service.load_movies()

        user_ratings = ratings[ratings["user_id"] == user_id].merge(movies[["movie_id", "title", "genres", "genres_list"]], on="movie_id")
        user_ratings = user_ratings.sort_values(["rating", "movie_id"], ascending=[False, True]).reset_index(drop=True)
        return [
            {
                "movie_id": int(row["movie_id"]),
                "title": row["title"],
                "genres": row["genres_list"],
                "rating": float(row["rating"]),
            }
            for _, row in user_ratings.iterrows()
        ]

    def get_user_profile(self, user_id: int) -> dict[str, Any]:
        ratings = movie_service.load_ratings()
        movies = movie_service.load_movies()
        user_ratings = ratings[ratings["user_id"] == user_id].merge(movies[["movie_id", "title", "genres", "genres_list"]], on="movie_id")
        if user_ratings.empty:
            raise ValueError(f"User {user_id} has no rating history.")

        favorite_genres: list[str] = []
        genre_counts: dict[str, int] = {}
        for _, row in user_ratings.sort_values("rating", ascending=False).iterrows():
            for genre in row["genres_list"]:
                genre_counts[genre] = genre_counts.get(genre, 0) + int(row["rating"] * 10)
        for genre, _ in sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            favorite_genres.append(genre)

        return {
            "user_id": int(user_id),
            "rating_count": int(len(user_ratings)),
            "average_rating": float(user_ratings["rating"].mean()),
            "favorite_genres": favorite_genres,
        }


recommendation_service = RecommendationService()
