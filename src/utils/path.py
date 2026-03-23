import os
import shutil
from datetime import datetime
from pathlib import Path

from .logger import LOG_PATH

WORK_DIR = Path(os.getenv("WORK_DIR") or "work")


def save_log(script_name: str) -> None:
    log_dir = Path(os.getenv("LOG_DIR", "logs") or "logs")
    log_path = log_dir / f"{script_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_dir.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(
        LOG_PATH,
        f"logs/{script_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )

    print(f"Log saved to {log_path}")
