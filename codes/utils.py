"""
field_logger: field card, clue and guessed word through entire turn
spymaster_logger: ranking
"""
import logging

def setup_filelogger(logger_name, file_name, level, add_console=True):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    handler = logging.FileHandler(file_name)

    if add_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)

    logger.addHandler(handler)
    return logger


