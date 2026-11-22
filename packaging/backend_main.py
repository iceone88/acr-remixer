"""Entry point of the frozen backend (stagegen-backend.exe): the CLI."""
import sys

from stagegen.cli import main

if __name__ == "__main__":
    sys.exit(main())
