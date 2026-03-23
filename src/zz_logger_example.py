import time

from utils.logger import LOG_PATH, init_logging

logger = init_logging(__name__)


@logger.timed("Example Timer via decorator")
def example_function():
    for i in range(5):
        print(f"Processing item {i + 1}")
        time.sleep(0.5)
    return 0


def main():
    logger.debug(f"This is a debug message and will only be shown in {LOG_PATH}")
    logger.info(
        "This is an info message and will be shown in both console and log file"
    )

    with logger.timer("Example Timer"):
        for i in range(5):
            print(f"Processing item {i + 1}")
            time.sleep(0.5)

    example_function()


if __name__ == "__main__":
    main()
