import pytest


@pytest.fixture
def calibrator(request):
    return request.config.getoption("--cal")

def test_new_cal_one(calibrator):
    if calibrator == "directbayes":
        assert 1 == 2
    elif calibrator == "mlbayeswoodbury":
        assert 3 == 3
    else:
        assert "Please pass --cal=x" == ""
