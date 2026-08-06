import json
from argparse import Namespace
from pathlib import Path


def load_config(path=None):
    config_path = Path(path) if path else Path(__file__).parents[1] / "config" / "default.json"
    values = json.loads(config_path.read_text(encoding="utf-8"))
    return Namespace(**values)
