import pandas as pd
# pyrefly: ignore [missing-import]
from fastapi import APIRouter, HTTPException, Query

from config import POPULAR_MIN_RATINGS, TOP_RATED_MIN_RATINGS
from services.recommendation_service import recommendation_service
from services.movie_service import movie_service

router = APIRouter(prefix="/movies", tags=["movies"])


@router.get("/genres", summary="List all available movie genres")
def get_genres() -> list[str]:
    return movie_service.get_genres()


@router.get("/popular", summary="Get popular movies by weighted rating")
def get_popular_movies(limit: int = Query(default=10, ge=1, le=50)) -> dict:
    try:
        movies = recommendation_service.get_popular_movies(limit=limit)
        return {
            "source": "weighted_rating",
            "minimum_rating_count": POPULAR_MIN_RATINGS,
            "count": int(len(movies)),
            "movies": [
                {
                    "movie_id": int(row["movie_id"]),
                    "title": row["title"],
                    "genres": row["genres_list"],
                    "average_rating": float(row["average_rating"]),
                    "rating_count": int(row["rating_count"]),
                    "weighted_rating": float(row["weighted_rating"]),
                }
                for _, row in movies.iterrows()
            ],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/top-rated", summary="Get top rated movies")
def get_top_rated_movies(limit: int = Query(default=10, ge=1, le=50)) -> dict:
    try:
        movies = recommendation_service.get_top_rated_movies(limit=limit, minimum_ratings=TOP_RATED_MIN_RATINGS)
        return {
            "source": "average_rating",
            "minimum_rating_count": TOP_RATED_MIN_RATINGS,
            "count": int(len(movies)),
            "movies": [
                {
                    "movie_id": int(row["movie_id"]),
                    "title": row["title"],
                    "genres": row["genres_list"],
                    "average_rating": float(row["average_rating"]),
                    "rating_count": int(row["rating_count"]),
                }
                for _, row in movies.iterrows()
            ],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/genre/{genre}", summary="Get the best movies in a genre sorted by average rating")
def get_genre_movies(genre: str, limit: int = Query(default=20, ge=1, le=100)) -> dict:
    movies = movie_service.load_movies()
    ratings = movie_service.load_ratings()
    normalized = genre.strip()

    filtered = movies[movies["genres_list"].apply(lambda value: normalized in value)]
    if filtered.empty:
        raise HTTPException(status_code=404, detail=f"Genre '{genre}' not found.")

    # Join with rating stats so we can sort by average rating
    rating_stats = (
        ratings.groupby("movie_id")["rating"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"count": "rating_count", "mean": "average_rating"})
    )
    merged = filtered.merge(rating_stats, on="movie_id", how="left")
    merged["rating_count"] = merged["rating_count"].fillna(0).astype(int)
    merged["average_rating"] = merged["average_rating"].fillna(0.0)

    # Sort: movies with ≥5 ratings first by avg rating desc, then unrated movies at end
    rated = merged[merged["rating_count"] >= 5].sort_values(
        ["average_rating", "rating_count"], ascending=[False, False]
    )
    unrated = merged[merged["rating_count"] < 5].sort_values("title")
    result = pd.concat([rated, unrated]).head(limit)

    return {
        "genre": normalized,
        "count": int(len(result)),
        "movies": [
            {
                "movie_id": int(row["movie_id"]),
                "title": row["title"],
                "genres": row["genres_list"],
                "average_rating": round(float(row["average_rating"]), 3) if row["rating_count"] >= 5 else None,
                "rating_count": int(row["rating_count"]),
            }
            for _, row in result.iterrows()
        ],
    }


@router.get("/search", summary="Search for movies by title")
def search_movies(q: str = Query(default="", min_length=1), limit: int = Query(default=10, ge=1, le=20)) -> dict:
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")
    results = recommendation_service.search_movies(q, limit=limit)
    return {"query": q, "count": len(results), "movies": results}


@router.get("/{movie_id}", summary="Get details for a movie")
def get_movie_details(movie_id: int) -> dict:
    movies = movie_service.load_movies()
    ratings = movie_service.load_ratings()
    movie = movies[movies["movie_id"] == movie_id]
    if movie.empty:
        raise HTTPException(status_code=404, detail="Movie not found.")
    movie_row = movie.iloc[0]
    ratings_for_movie = ratings[ratings["movie_id"] == movie_id]
    average = float(ratings_for_movie["rating"].mean()) if not ratings_for_movie.empty else 0.0
    return {
        "movie_id": int(movie_row["movie_id"]),
        "title": movie_row["title"],
        "genres": movie_row["genres_list"],
        "average_rating": average,
        "rating_count": int(len(ratings_for_movie)),
    }
