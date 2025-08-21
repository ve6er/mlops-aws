import os
import sys
import pandas as pd
import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Add src folder to sys.path so tests can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# Mock nltk.download to avoid downloads in CI
import nltk
# nltk.download = lambda *args, **kwargs: None
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Imports from src/
from src import data_ingestion as di
from src import data_preprocessing as dp
from src import feature_engineering as fe
from src import model_building as mb
from src import model_evaluation as me


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "v1": ["spam", "ham"],
        "v2": ["Win money now", "Hello friend"],
        "Unnamed: 2": [None, None],
        "Unnamed: 3": [None, None],
        "Unnamed: 4": [None, None]
    })


@pytest.fixture
def processed_df():
    return pd.DataFrame({
        "text": ["win money", "hello friend"],
        "target": [1, 0]
    })


def test_preprocess_data(sample_df):
    df = di.preprocess_data(sample_df.copy())
    assert "target" in df.columns
    assert "text" in df.columns
    assert df.shape[0] == 2


def test_transform_text():
    text = "Hello, This is a TEST!"
    result = dp.transform_text(text)
    assert isinstance(result, str)
    assert result != ""


def test_preprocess_df(processed_df):
    df = dp.preprocess_df(processed_df.copy(), text_column="text", target_column="target")
    assert "text" in df.columns
    assert "target" in df.columns


def test_apply_tfidf(processed_df):
    train_df, test_df = fe.apply_tfidf(processed_df, processed_df, max_features=5)
    assert "label" in train_df.columns
    assert train_df.shape[0] == processed_df.shape[0]


def test_train_model():
    X = np.array([[0, 1], [1, 0]])
    y = np.array([0, 1])
    params = {"n_estimators": 5, "random_state": 42}
    model = mb.train_model(X, y, params)
    assert isinstance(model, RandomForestClassifier)


def test_evaluate_model():
    X = np.array([[0, 1], [1, 0]])
    y = np.array([0, 1])
    clf = RandomForestClassifier().fit(X, y)
    metrics = me.evaluate_model(clf, X, y)
    assert all(k in metrics for k in ["accuracy", "precision", "recall", "auc"])
