from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from code_agent.main import app


if __name__ == "__main__":
    app()
