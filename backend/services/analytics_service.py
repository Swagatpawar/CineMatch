from __future__ import annotations

import pandas as pd

from config import POPULAR_MIN_RATINGS, TOP_RATED_MIN_RATINGS
from services.movie_service import movie_service


class AnalyticsService:
    def get_model_metrics(self) -> dict:
        return {
            "models": [
                {"name": "User-Based Pearson CF", "rmse": 0.9299},
                {"name": "Item-Based Pearson CF", "rmse": 1.0956},
                {"name": "SVD", "rmse": 0.8706},
            ],
            "precision_at_5": 0.1028,
            "precision_at_10": 0.0919,
            "best_model": "SVD",
            "best_factors": 50,
        }

    def get_popularity_stats(self) -> pd.DataFrame:
        ratings = movie_service.load_ratings()
        movies = movie_service.load_movies()

        rating_summary = ratings.groupby("movie_id")["rating"].agg(["count", "mean"]).reset_index()
        rating_summary = rating_summary.rename(columns={"count": "rating_count", "mean": "average_rating"})
        rating_summary["weighted_rating"] = (
            rating_summary["average_rating"] * rating_summary["rating_count"]
        ) / (rating_summary["rating_count"] + POPULAR_MIN_RATINGS)
        merged = rating_summary.merge(movies[["movie_id", "title", "genres", "genres_list"]], on="movie_id", how="left")
        return merged.sort_values(["weighted_rating", "rating_count"], ascending=[False, False]).reset_index(drop=True)

    def get_top_rated(self) -> pd.DataFrame:
        ratings = movie_service.load_ratings()
        movies = movie_service.load_movies()

        rating_summary = ratings.groupby("movie_id")["rating"].agg(["count", "mean"]).reset_index()
        rating_summary = rating_summary.rename(columns={"count": "rating_count", "mean": "average_rating"})
        filtered = rating_summary[rating_summary["rating_count"] >= TOP_RATED_MIN_RATINGS]
        filtered = filtered.merge(movies[["movie_id", "title", "genres", "genres_list"]], on="movie_id", how="left")
        return filtered.sort_values(["average_rating", "rating_count"], ascending=[False, False]).reset_index(drop=True)


analytics_service = AnalyticsService()
