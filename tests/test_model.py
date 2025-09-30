# tests/test_model.py
import pytest
import numpy as np
from models.model import train_model

def test_train_model_output_shape():
    X = np.random.rand(100, 10)  # fake features (100 rows, 10 cols)
    y = np.random.randint(0, 2, 100)  # fake labels (0 or 1)
    
    model = train_model(X, y)  # train your model
    
    # Check: prediction shape must equal y shape
    assert model.predict(X).shape == y.shape

def test_train_model_type():
    X = np.random.rand(50, 5)
    y = np.random.randint(0, 2, 50)
    
    model = train_model(X, y)
    
    # Check: model must have a 'predict' function
    assert hasattr(model, "predict")
