#!/usr/bin/env python

if __name__ == "__main__":
    import sys
    assert sys.version_info >= (3, 8), "Python 3.8+ required, got: {}".format(sys.version)

    from lazy_transpiler.main import main
    main()
