from fastapi import APIRouter, HTTPException, Query

from config import DEFAULT_RECOMMENDATION_LIMIT
from services.recommendation_service import recommendation_service

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


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
