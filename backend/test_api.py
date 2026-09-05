import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_genres_endpoint():
    response = client.get('/api/genres')
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert 'Action' in response.json()


def test_popular_movies_endpoint():
    response = client.get('/api/movies/popular')
    assert response.status_code == 200
    data = response.json()
    assert data['count'] >= 1
    assert 'movies' in data


def test_top_rated_movies_endpoint():
    response = client.get('/api/movies/top-rated')
    assert response.status_code == 200
    data = response.json()
    assert data['count'] >= 1
    assert 'movies' in data


def test_search_movies_endpoint():
    response = client.get('/api/movies/search', params={'q': 'Matrix'})
    assert response.status_code == 200
    assert response.json()['count'] >= 1


def test_existing_user_recommendations():
    response = client.get('/api/recommendations/1')
    assert response.status_code == 200
    data = response.json()
    assert data['type'] in {'personalized', 'cold_start'}
    assert 'recommendations' in data


def test_cold_start_recommendations():
    response = client.get('/api/recommendations/999999')
    assert response.status_code == 200
    data = response.json()
    assert data['type'] == 'cold_start'


def test_model_metrics_endpoint():
    response = client.get('/api/analytics/model-performance')
    assert response.status_code == 200
    data = response.json()
    assert data['best_model'] == 'SVD'
    assert data['models'][2]['name'] == 'SVD'


def test_cold_start_with_ratings_returns_recommendations():
    response = client.post('/api/recommendations/cold-start', json={
        'genres': ['Action', 'Sci-Fi', 'Thriller'],
        'ratings': [
            {'movie_id': 2571, 'rating': 5},
            {'movie_id': 1196, 'rating': 4},
            {'movie_id': 260, 'rating': 5},
            {'movie_id': 1210, 'rating': 4},
            {'movie_id': 589, 'rating': 4},
        ],
    })
    assert response.status_code == 200
    data = response.json()
    assert data['user_type'] == 'new'
    assert len(data['recommendations']) == 5
    assert data['recommendations'][0]['reason']


def test_cold_start_fallback_without_preferences_returns_movies():
    response = client.post('/api/recommendations/cold-start', json={'genres': [], 'ratings': []})
    assert response.status_code == 200
    assert response.json()['recommendations']


def test_cold_start_rejects_invalid_movie_and_rating():
    unknown_movie = client.post('/api/recommendations/cold-start', json={
        'genres': ['Drama'], 'ratings': [{'movie_id': 999999, 'rating': 5}],
    })
    invalid_rating = client.post('/api/recommendations/cold-start', json={
        'genres': ['Drama'], 'ratings': [{'movie_id': 1, 'rating': 6}],
    })
    assert unknown_movie.status_code == 400
    assert invalid_rating.status_code == 422


def test_existing_user_route_remains_svd_compatible():
    response = client.get('/api/recommendations/101')
    assert response.status_code == 200
    assert response.json()['recommendations']
