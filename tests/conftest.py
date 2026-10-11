"""Synthetic CSV inputs shared with the standalone smoke check."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from pipeline.smoke_test import synthetic_sources


@pytest.fixture
def source_paths(tmp_path):
    return synthetic_sources(tmp_path)
