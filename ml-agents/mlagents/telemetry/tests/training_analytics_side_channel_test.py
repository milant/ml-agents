import pytest


@pytest.mark.usefixtures("simple_run_options")
def test_default_settings(simple_run_options):
    assert "test1" in simple_run_options.behaviors
