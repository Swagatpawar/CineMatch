from __future__ import annotations

from typing import Any

import pandas as pd

from config import DEFAULT_RECOMMENDATION_LIMIT, MAX_RECOMMENDATION_LIMIT, POPULAR_MIN_RATINGS
from services.model_service import model_service
from services.movie_service import movie_service


class RecommendationService:
    SUPPORTED_GENRES = {
        "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
        "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
        "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
    }

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

    def get_cold_start_recommendations(
        self,
        genres: list[str],
        ratings: list[dict[str, float | int]],
        limit: int = DEFAULT_RECOMMENDATION_LIMIT,
    ) -> dict[str, Any]:
        """Rank new-user recommendations without inventing an SVD user vector."""
        if not 1 <= limit <= 20:
            raise ValueError("limit must be between 1 and 20.")

        movies = movie_service.load_movies()
        existing_ratings = movie_service.load_ratings()
        normalized_genres = list(dict.fromkeys(genre.strip() for genre in genres if genre.strip()))
        invalid_genres = sorted(set(normalized_genres) - self.SUPPORTED_GENRES)
        if invalid_genres:
            raise ValueError(f"Unsupported genre(s): {', '.join(invalid_genres)}")

        movie_ids = set(movies["movie_id"].astype(int))
        rating_map: dict[int, float] = {}
        for entry in ratings:
            movie_id = int(entry["movie_id"])
            rating = float(entry["rating"])
            if movie_id not in movie_ids:
                raise ValueError(f"Movie ID {movie_id} does not exist.")
            if not 1 <= rating <= 5:
                raise ValueError("Ratings must be between 1 and 5.")
            rating_map[movie_id] = rating

        stats = (
            existing_ratings.groupby("movie_id")["rating"]
            .agg(rating_count="count", average_rating="mean")
            .reset_index()
        )
        candidates = movies.merge(stats, on="movie_id", how="left")
        candidates["rating_count"] = candidates["rating_count"].fillna(0)
        candidates["average_rating"] = candidates["average_rating"].fillna(0)
        candidates = candidates[~candidates["movie_id"].isin(rating_map)]

        genre_set = set(normalized_genres)
        candidates["genre_match"] = candidates["genres_list"].apply(
            lambda values: len(genre_set.intersection(values)) if genre_set else 0
        )
        candidates["popularity"] = (
            candidates["average_rating"] * candidates["rating_count"]
            / (candidates["rating_count"] + POPULAR_MIN_RATINGS)
        )
        candidates["score"] = candidates["popularity"] + candidates["genre_match"] * 0.35
        method = "popularity_fallback"
        reason = "Popular and highly rated movies from the catalog."

        if rating_map:
            rated_ids = list(rating_map)
            overlap = existing_ratings[existing_ratings["movie_id"].isin(rated_ids)].copy()
            overlap["new_rating"] = overlap["movie_id"].map(rating_map)
            overlap["new_centered"] = overlap["new_rating"] - sum(rating_map.values()) / len(rating_map)
            user_means = overlap.groupby("user_id")["rating"].transform("mean")
            overlap["existing_centered"] = overlap["rating"] - user_means
            overlap["product"] = overlap["new_centered"] * overlap["existing_centered"]
            similarity = overlap.groupby("user_id").agg(
                overlap_count=("movie_id", "count"),
                numerator=("product", "sum"),
                existing_norm=("existing_centered", lambda values: float((values ** 2).sum()) ** 0.5),
            )
            new_norm = float((overlap.drop_duplicates("movie_id")["new_centered"] ** 2).sum()) ** 0.5
            similarity["similarity"] = similarity["numerator"] / (new_norm * similarity["existing_norm"] + 1e-9)
            similarity = similarity[(similarity["overlap_count"] >= 2) & (similarity["similarity"] > 0)]
            similar_users = similarity.sort_values("similarity", ascending=False).head(50)

            if not similar_users.empty:
                liked = existing_ratings[existing_ratings["user_id"].isin(similar_users.index)].merge(
                    similar_users[["similarity"]], left_on="user_id", right_index=True
                )
                liked = liked[liked["rating"] >= 3.5]
                liked_scores = liked.groupby("movie_id").apply(
                    lambda rows: float((rows["rating"] * rows["similarity"]).sum() / rows["similarity"].abs().sum()),
                    include_groups=False,
                )
                candidates["similarity_score"] = candidates["movie_id"].map(liked_scores).fillna(0)
                candidates["score"] += candidates["similarity_score"] * 0.75
                method = "similar_users_and_genres"
                reason = "Similar MovieLens users with tastes like yours enjoyed this."
            elif normalized_genres:
                method = "genre_and_popularity"
                reason = "Highly rated among movies matching your preferences."
        elif normalized_genres:
            method = "genre_and_popularity"
            reason = "Matches your selected genres and has strong MovieLens ratings."

        candidates = candidates.sort_values(
            ["score", "genre_match", "rating_count"], ascending=[False, False, False]
        ).head(limit)
        recommendations = []
        for _, row in candidates.iterrows():
            row_genres = set(row["genres_list"])
            if genre_set and row_genres.intersection(genre_set):
                row_reason = reason
            elif method == "similar_users_and_genres":
                row_reason = "Similar users enjoyed this movie."
            else:
                row_reason = "Popular and highly rated in MovieLens."
            recommendations.append({
                "movie_id": int(row["movie_id"]),
                "title": row["title"],
                "genres": row["genres_list"],
                "score": round(float(row["score"]), 3),
                "reason": row_reason,
            })

        return {
            "user_type": "new",
            "method": "cold_start",
            "recommendation_method": method,
            "recommendations": recommendations,
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
