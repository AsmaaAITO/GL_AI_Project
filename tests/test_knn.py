import sys
import os
import pytest
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.knn import KNNModel

def test_knn_initialization():
    """Test KNNModel initialization."""
    model = KNNModel(n_neighbors=3)
    assert model.n_neighbors == 3
    assert not model.is_fitted

def test_knn_fit_predict():
    """Test KNNModel fit and predict."""
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.array([0, 0, 1, 1])
    
    model = KNNModel(n_neighbors=1)
    model.fit(X, y)
    
    assert model.is_fitted
    
    # Predict on training data
    predictions = model.predict(X)
    assert len(predictions) == 4
    assert np.array_equal(predictions, y)
    
    # Test predict_proba
    probs = model.predict_proba(X)
    assert probs.shape == (4, 2)

if __name__ == "__main__":
    test_knn_initialization()
    test_knn_fit_predict()
    print("All KNN tests passed!")
