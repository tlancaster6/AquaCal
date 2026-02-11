"""Pytest fixtures for synthetic pipeline tests."""

import pytest
from .ground_truth import create_scenario, SyntheticScenario


@pytest.fixture(scope="class")
def scenario_ideal() -> SyntheticScenario:
    """Ideal scenario: no noise, verify math."""
    return create_scenario("ideal")


@pytest.fixture(scope="class")
def scenario_minimal() -> SyntheticScenario:
    """Minimal scenario: 2 cameras, edge case."""
    return create_scenario("minimal")


@pytest.fixture(scope="class")
def scenario_realistic() -> SyntheticScenario:
    """Realistic scenario: 13 cameras matching actual hardware."""
    return create_scenario("realistic")
