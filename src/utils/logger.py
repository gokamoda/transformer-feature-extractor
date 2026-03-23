import inspect
import logging
import os
import time
from datetime import datetime
from functools import partial, wraps
from pathlib import Path

from pytz import timezone
from rich.logging import RichHandler
from tqdm import tqdm as std_tqdm

LOG_PATH = "latest.log"
tqdm = partial(std_tqdm, dynamic_ncols=True)


def _custom_time(*args):
    return datetime.now(timezone("Asia/Tokyo")).timetuple()


class Timer:
    def __init__(self, logger, label):
        self.logger = logger
        self.label = label

    def __enter__(self):
        self.logger.info(f"Start stopwatch: {self.label}")
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.perf_counter() - self.start

        # Determine appropriate unit
        if self.elapsed < 1:
            time_value = self.elapsed * 1000
            unit = "ms"
        elif self.elapsed < 60:
            time_value = self.elapsed
            unit = "s"
        elif self.elapsed < 3600:
            time_value = self.elapsed / 60
            unit = "min"
        else:
            time_value = self.elapsed / 3600
            unit = "h"

        frame = inspect.currentframe()
        outer_frames = inspect.getouterframes(frame)
        # [0]=__exit__, [1]=wrapper, [2]=caller

        # Find the first "real" caller frame
        for outer in outer_frames:
            filename = outer.filename
            filename_short = os.path.relpath(filename, os.getcwd())
            # Customize the filtering as needed
            if not any(
                skip in filename
                for skip in ["logger.py", "contextlib", "ipykernel", "asyncio", "runpy"]
            ):
                lineno = outer.lineno
                break
        # filename = outer_frame.filename
        # lineno = outer_frame.lineno

        self.logger.info(
            f"{self.label} took {time_value:.2f} {unit}\nat {filename_short}:{lineno}"
        )


class CustomLogger(logging.Logger):
    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)
        self._warned_once = set()

    def warn_once(self, msg: str):
        if msg not in self._warned_once:
            self._warned_once.add(msg)
            self.warning(msg)

    def timer(self, label: str):
        """Use `with` statement to measure elapsed time."""
        return Timer(self, label)

    def timed(self, label: str | None = None):
        """"""

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                nonlocal label
                label = label or func.__name__
                with self.timer(label):
                    return func(*args, **kwargs)

            return wrapper

        return decorator


def init_logging(
    logger_name: str, log_path: str = LOG_PATH, clear=False
) -> CustomLogger:
    """_summary_

    Parameters
    ----------
    logger_name : str
        _description_
    log_path : str, optional
        _description_, by default "logs/info.log"

    Returns
    -------
    logging.Logger
        _description_
    """

    logging.setLoggerClass(CustomLogger)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    path = Path(log_path)

    dir_path = path.parent
    if not dir_path.exists():
        dir_path.mkdir(parents=True)

    if clear:
        with open(log_path, "w") as f:
            f.write("")

    if logger.handlers == []:
        dir_path.mkdir(parents=True, exist_ok=True)
        formatter = logging.Formatter(
            "\n%(asctime)s/%(levelname)s/%(name)s/%(funcName)s():%(lineno)s\n"
            "%(message)s"
        )

        formatter.converter = _custom_time

        # file handler
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)

        # console handler
        ch = RichHandler(rich_tracebacks=True, show_time=False, show_level=False)
        # ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

        logger.propagate = False

    return logger  # ty: ignore[invalid-return-type]
