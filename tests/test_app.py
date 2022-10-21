from api.app import home


def test_home():
    res = home()
    assert res == "Homepage"


def test_extract_entities():
    pass
