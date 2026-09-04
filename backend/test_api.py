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
