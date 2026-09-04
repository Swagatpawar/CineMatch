from fastapi import APIRouter, HTTPException

from services.recommendation_service import recommendation_service

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/{user_id}", summary="Get user summary and profile")
def get_user(user_id: int) -> dict:
    try:
        return recommendation_service.get_user_profile(user_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{user_id}/history", summary="Get the user's rating history")
def get_user_history(user_id: int) -> dict:
    try:
        history = recommendation_service.get_user_history(user_id)
        return {"user_id": user_id, "count": len(history), "ratings": history}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
