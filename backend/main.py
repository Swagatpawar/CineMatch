from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.analytics import router as analytics_router
from api.movies import router as movies_router
from api.recommendations import router as recommendations_router
from api.users import router as users_router
from config import ALLOWED_ORIGINS
from services.model_service import model_service
from services.movie_service import movie_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model_service.initialize()
        yield
    except Exception as exc:  # pragma: no cover - defensive initialization guard
        raise RuntimeError(f"CineMatch backend initialization failed: {exc}") from exc


app = FastAPI(
    title="CineMatch",
    description="Intelligent movie recommendation engine powered by collaborative filtering and SVD.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(recommendations_router, prefix="/api")
app.include_router(movies_router, prefix="/api")
app.include_router(users_router, prefix="/api")
app.include_router(analytics_router, prefix="/api")


@app.get("/api/genres")
def get_genres() -> list[str]:
    return movie_service.get_genres()


@app.get("/health")
def health_check() -> dict[str, str | bool]:
    try:
        model_service.get_model()
        movie_service.load_movies()
        movie_service.load_ratings()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="CineMatch dependencies are not ready.") from exc

    return {
        "status": "ok",
        "service": "CineMatch",
        "model_loaded": model_service.model is not None,
        "data_loaded": movie_service.movies_df is not None and movie_service.ratings_df is not None,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
