from core.rng import derive_rng


def test_same_keys_same_stream():
    r1 = derive_rng(42, "arrival", "P0001", 3)
    r2 = derive_rng(42, "arrival", "P0001", 3)
    assert [r1.random() for _ in range(5)] == [r2.random() for _ in range(5)]


def test_different_keys_differ():
    r1 = derive_rng(42, "arrival", "P0001", 3)
    r2 = derive_rng(42, "arrival", "P0001", 4)
    assert [r1.random() for _ in range(3)] != [r2.random() for _ in range(3)]


def test_different_channels_differ():
    r1 = derive_rng(42, "arrival", "P0001", 3)
    r2 = derive_rng(42, "anomaly_time", "P0001", 3)
    assert [r1.random() for _ in range(3)] != [r2.random() for _ in range(3)]
