"""PyInstaller entry point for the ACR Remixer StageGen command-line tool."""
from stagegen.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
