def test_genetic_algorithm_importable():
    import genetic_algorithm

    assert hasattr(genetic_algorithm, "Config")
    assert hasattr(genetic_algorithm, "Result")
