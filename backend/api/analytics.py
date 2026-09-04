from fastapi import APIRouter

from services.analytics_service import analytics_service

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/model-performance", summary="Return model evaluation metrics")
def get_model_performance() -> dict:
    return analytics_service.get_model_metrics()
