def test_suite():
    """Returns unittest.TestSuite for this package"""
    import unittest
    from os.path import dirname
    basedir = dirname(dirname(__file__))
    # print(desispec_dir)
    return unittest.defaultTestLoader.discover(basedir,
        top_level_dir=dirname(basedir))

