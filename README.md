# CineMatch

CineMatch is a MovieLens movie recommendation application. It combines a FastAPI service with a React/Vite frontend and a local NumPy SVD model.

## Features

- Personalized top-five recommendations for MovieLens users
- Personalized recommendations within a selected genre
- Popular and top-rated movie discovery
- Browse the canonical MovieLens genres
- Movie search and details
- User rating statistics and history
- Model performance metrics
- Cold-start fallback to popularity-based recommendations
- Responsive dark cinematic interface

## Stack

- Backend: Python, FastAPI, pandas, NumPy, scikit-learn
- Frontend: React 19, TypeScript, Vite
- Model: custom NumPy latent-factor SVD implementation
- Dataset: MovieLens latest-small CSV data

## Repository Layout

```text
movie/
├── backend/
│   ├── api/                    FastAPI routers
│   ├── data/                   local runtime CSV data (ignored)
│   ├── models/                 local trained model (ignored)
│   ├── services/               data, recommendation, and analytics services
│   ├── build_model.py          build the local SVD artifact
│   ├── cinematch_model.py      SVD model and prediction types
│   ├── config.py               paths and environment configuration
│   ├── main.py                 FastAPI application
│   ├── prepare_data.py         download/prepare MovieLens data and model
│   ├── requirements.txt
│   └── test_api.py
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── vite.config.ts
├── .gitignore
└── README.md
```

The application reads runtime data from `backend/data/movies.csv` and `backend/data/ratings.csv`. It reads the trained artifact from `backend/models/cinematch_svd_final.pkl`. These generated and downloaded files are intentionally ignored by Git.

## Backend Setup

Use Python 3.12 or a compatible current Python version:

```powershell
cd backend
python -m venv ..\.venv
..\.venv\Scripts\activate
pip install -r requirements.txt
```

Prepare the local MovieLens data and model. This downloads the MovieLens latest-small archive and creates the runtime CSV files and SVD artifact locally:

```powershell
python prepare_data.py
```

The preparation script uses the MovieLens `movies.csv` and `ratings.csv` files. No MovieLens data, model pickle, or downloaded archive is committed to this repository.

Start FastAPI from the repository root or from `backend`:

```powershell
cd backend
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Available backend configuration variables are optional:

- `HOST`, default `0.0.0.0`
- `PORT`, default `8000`
- `ALLOWED_ORIGINS`, comma-separated frontend origins

## Frontend Setup

```powershell
cd frontend
npm install
npm run dev
```

The Vite development server normally runs at `http://localhost:5173`. The frontend calls the FastAPI service at `http://localhost:8000`.

For a production build:

```powershell
cd frontend
npm run build
npm run preview
```

## API

- `GET /health`
- `GET /docs`
- `GET /api/genres`
- `GET /api/recommendations/{user_id}`
- `GET /api/recommendations/{user_id}/genre/{genre}`
- `GET /api/movies/popular`
- `GET /api/movies/top-rated`
- `GET /api/movies/genre/{genre}`
- `GET /api/movies/search?q={query}`
- `GET /api/movies/{movie_id}`
- `GET /api/users/{user_id}`
- `GET /api/users/{user_id}/history`
- `GET /api/analytics/model-performance`

## ML Methodology

The preparation workflow loads MovieLens ratings, builds a sparse user-movie utility matrix, centers observed ratings by user and movie means, and applies NumPy SVD. The trained latent factors are saved as `backend/models/cinematch_svd_final.pkl`. Recommendations score unseen movies, exclude watched movies, and return the highest predicted ratings. Unknown or sparse users receive a popularity-based cold-start fallback.

The current implementation uses 50 latent factors. It is a custom NumPy SVD model and does not use Surprise-style `n_epochs`, `lr_all`, or `reg_all` parameters.

## Evaluation Metrics

These are the completed evaluation values exposed by the analytics endpoint:

| Model | RMSE |
| --- | ---: |
| User-Based Pearson CF | 0.9299 |
| Item-Based Pearson CF | 1.0956 |
| SVD | 0.8706 |

- SVD Precision@5: `0.1028`
- SVD Precision@10: `0.0919`
- Best SVD factors: `50`

Lower RMSE is better. SVD is the best-performing evaluated model.

## Git and Local Artifacts

The root `.gitignore` excludes virtual environments, dependency folders, environment files, model artifacts, downloaded datasets, CSV/DAT files, caches, logs, and build output. These files remain available locally and are recreated or supplied during setup.

The legacy local directory `backend/model/` is not used by the application and is ignored. The runtime locations are `backend/data/` and `backend/models/`.

Before committing, inspect the result carefully:

```powershell
git status --short
git diff --cached
```

Do not commit secrets, `.env` files, virtual environments, downloaded datasets, or model binaries.
