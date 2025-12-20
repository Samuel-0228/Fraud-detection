import pytest
from src.preprocess import ip_to_int


def test_ip_to_int():
    assert ip_to_int('192.168.1.1') == 3232235777
    assert ip_to_int('invalid') is None  # Handles errors

# Run with pytest
