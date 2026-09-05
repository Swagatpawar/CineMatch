from pydantic import BaseModel, Field, field_validator
from fastapi import APIRouter, HTTPException, Query

from config import DEFAULT_RECOMMENDATION_LIMIT
from services.recommendation_service import recommendation_service

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


class ColdStartRating(BaseModel):
    movie_id: int = Field(gt=0)
    rating: float = Field(ge=1, le=5)


class ColdStartRequest(BaseModel):
    genres: list[str] = Field(default_factory=list, max_length=18)
    ratings: list[ColdStartRating] = Field(default_factory=list, max_length=20)
    limit: int = Field(default=5, ge=1, le=20)

    @field_validator("genres")
    @classmethod
    def normalize_genres(cls, genres: list[str]) -> list[str]:
        return list(dict.fromkeys(genre.strip() for genre in genres if genre.strip()))


@router.post("/cold-start", summary="Get recommendations for a new CineMatch user")
def get_cold_start_recommendations(request: ColdStartRequest) -> dict:
    try:
        return recommendation_service.get_cold_start_recommendations(
            genres=request.genres,
            ratings=[rating.model_dump() for rating in request.ratings],
            limit=request.limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{user_id}", summary="Get personalized recommendations for a user")
def get_recommendations_for_user(
    user_id: int,
    limit: int = Query(default=DEFAULT_RECOMMENDATION_LIMIT, ge=1, le=50),
) -> dict:
    try:
        return recommendation_service.get_recommendations_for_user(user_id, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{user_id}/genre/{genre}", summary="Get personalized recommendations within a genre")
def get_genre_recommendations(
    user_id: int,
    genre: str,
    limit: int = Query(default=DEFAULT_RECOMMENDATION_LIMIT, ge=1, le=50),
) -> dict:
    try:
        return recommendation_service.get_genre_recommendations(user_id, genre, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
