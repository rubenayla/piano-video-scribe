"""Fixtures for test1: run the pipeline once, share results across tests."""
import pytest
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from run_test import load_expected, run_pipeline, VIDEO


@pytest.fixture(scope="module")
def expected():
    return load_expected()


@pytest.fixture(scope="module")
def pipeline_result(expected):
    kb = expected["keyboard"]
    notes, white_keys, black_keys = run_pipeline(
        VIDEO, None, expected["bpm"], kb["frame"])
    return notes, white_keys, black_keys


@pytest.fixture(scope="module")
def notes(pipeline_result):
    return pipeline_result[0]


@pytest.fixture(scope="module")
def white_keys(pipeline_result):
    return pipeline_result[1]


@pytest.fixture(scope="module")
def black_keys(pipeline_result):
    return pipeline_result[2]
