"""Hub cache location: under the researcher's home, never beside the code."""

from pathlib import Path

from nomikos_inference.hub.cache import DEFAULT_CACHE_ROOT


def test_the_hub_cache_defaults_under_the_researchers_home_directory() -> None:
    """In a wheel, "beside the code" is site-packages."""
    assert DEFAULT_CACHE_ROOT.is_relative_to(Path.home())
