import pytest


@pytest.fixture
def emulator(request):
    return request.config.getoption("--emu")

# TODO: If I uncomment this, --cmdopt1 doesn't seem to have an effect.
#@pytest.mark.parametrize(
#    "emulator", ["PCGP", "PCSK"]
#)
def test_new_emu_uno(emulator):
    if emulator == "PCGP":
        assert 1 == 2
    elif emulator == "PCSK":
        assert 3 == 3
    else:
        assert "Please pass --emu=x" == ""
