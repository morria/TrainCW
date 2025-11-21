"""Test basic package imports and metadata."""

import pytest


@pytest.mark.unit
def test_import_package():
    """Test that the traincw package can be imported."""
    import traincw

    assert traincw is not None


@pytest.mark.unit
def test_package_version():
    """Test that package version is defined."""
    from traincw import __version__

    assert __version__ is not None
    assert isinstance(__version__, str)
    assert len(__version__) > 0


@pytest.mark.unit
def test_package_metadata():
    """Test that package metadata is accessible."""
    from traincw import __author__, __version__

    assert __version__ == "0.1.0"
    assert __author__ is not None
    assert isinstance(__author__, str)
