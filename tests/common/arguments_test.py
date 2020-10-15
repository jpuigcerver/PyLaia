from laia.common.arguments import get_key


def test_get_key():
    assert get_key({"learning_rate": 1.0}, "learning_rate") == 1.0
    assert get_key({}, "learning_rate") == 5e-4
    assert not get_key({"foo": "bar"}, "nesterov")
    assert get_key({"foo": "bar", "nesterov": True}, "nesterov")
