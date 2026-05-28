"""Make the src-layout package importable in tests without installation."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
