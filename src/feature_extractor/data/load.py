import json
from pathlib import Path


def load_jsonl_text_dataset(file_path: str | Path) -> list[dict[str, str]]:
    """Load a JSONL dataset"""

    data = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if "text" not in item:
                raise ValueError(
                    f"Each JSON object must have a 'text' field. Found: {item}"
                )
            data.append(item)
    return data
