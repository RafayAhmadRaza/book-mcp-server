"""Console entry point.

Keeps ``python server.py`` working from the repository root while the real
implementation lives in the ``book_mcp`` package.
"""

from book_mcp.server import main

if __name__ == "__main__":
    main()
