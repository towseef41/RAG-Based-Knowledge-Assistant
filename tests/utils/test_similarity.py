import numpy as np
import pytest
from app.utils.similarity import cosine_similarity 

def test_identical_vectors():
    vec = [1, 2, 3]
    assert cosine_similarity(vec, vec) == pytest.approx(1.0)

def test_orthogonal_vectors():
    vec1 = [1, 0]
    vec2 = [0, 1]
    assert cosine_similarity(vec1, vec2) == pytest.approx(0.0)

def test_opposite_vectors():
    vec1 = [1, 0]
    vec2 = [-1, 0]
    assert cosine_similarity(vec1, vec2) == pytest.approx(-1.0)

def test_random_vectors():
    vec1 = [1, 2, 3]
    vec2 = [4, 5, 6]
    expected = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    assert cosine_similarity(vec1, vec2) == pytest.approx(expected)

def test_zero_vector_input():
    vec1 = [0, 0, 0]
    vec2 = [1, 2, 3]
    assert cosine_similarity(vec1, vec2) == 0.0
    assert cosine_similarity(vec2, vec1) == 0.0
    assert cosine_similarity(vec1, vec1) == 0.0

def test_different_lengths_should_fail():
    vec1 = [1, 2]
    vec2 = [1, 2, 3]
    with pytest.raises(ValueError):
        cosine_similarity(vec1, vec2)
