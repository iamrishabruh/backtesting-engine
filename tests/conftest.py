import os

import pytest


@pytest.fixture
def repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def sample_csv_path(repo_root):
    return os.path.join(repo_root, "examples", "sample_ohlcv.csv")
