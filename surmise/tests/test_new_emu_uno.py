import pytest


@pytest.fixture
def cmdopt1(request):
    return request.config.getoption("--cmdopt1")

# TODO: If I uncomment this, --cmdopt1 doesn't seem to have an effect.
#@pytest.mark.parametrize(
#    "cmdopt1", ["PCGP", "PCSK"]
#)
def test_new_emu_uno(cmdopt1):
    if cmdopt1 == "PCGP":
        assert 1 == 2
    elif cmdopt1 == "PCSK":
        assert 3 == 3
    else:
        assert "Please pass --cmdopt1=x" == ""
