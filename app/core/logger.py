import logging
from app.core.config import settings

LOG_LEVEL = settings.LOG_LEVEL.upper() if settings.LOG_LEVEL else "INFO"
level_mapping = logging.getLevelNamesMapping()


def setup_logger(
    name: str, log_file: str, level=level_mapping.get(LOG_LEVEL, logging.INFO)
) -> logging.Logger:
    """Set up a logger with the specified name, log file, and level."""
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


logger = setup_logger("app_logger", "app.log")
