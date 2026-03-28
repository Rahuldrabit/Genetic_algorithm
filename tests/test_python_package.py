def test_python_package_alias_importable():
    import genetic_algorithm

    assert hasattr(genetic_algorithm, "Config")

