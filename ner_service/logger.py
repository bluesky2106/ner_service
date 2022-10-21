import logging
import logging.config

LOGGER_NAME = "ner"
LEVEL = logging.INFO

logging.config.fileConfig(fname='logger_config.conf',
                          disable_existing_loggers=False)
_logger = logging.getLogger(LOGGER_NAME)
_logger.setLevel(LEVEL)


def get_logger():
    return _logger


if __name__ == "__main__":
    get_logger().debug("hello world")
