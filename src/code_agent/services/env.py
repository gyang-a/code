from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv as dotenv_load_dotenv


def load_dotenv(path: str | Path, *, override: bool = False) -> dict[str, str]:
    env_path = Path(path)
    if not env_path.exists() or not env_path.is_file():
        return {}

    loaded = {
        key: value
        for key, value in dotenv_values(env_path).items()
        if value is not None
    }
    dotenv_load_dotenv(env_path, override=override)
    return loaded
