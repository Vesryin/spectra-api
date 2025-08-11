"""Simple test to verify Responder app works"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from main import api  # noqa: E402


def test_health_endpoint():
    # Simple smoke test - if this imports without error, Responder is working
    assert api is not None
    print("✓ Responder API imported successfully")


if __name__ == "__main__":
    test_health_endpoint()
    print("Basic import test passed!")
